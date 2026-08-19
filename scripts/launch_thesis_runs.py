#!/usr/bin/env python
"""Lancia le run finali della tesi: sette bracci, un solo protocollo.

    python scripts/launch_thesis_runs.py --arms hybrid_soft --dry-run
    python scripts/launch_thesis_runs.py --arms all --seeds 1 2 3 4 5

Perche' un launcher nuovo invece di ``launch_grad_diagnostics.py``: quello
esporta gli iperparametri dallo studio Optuna leggendo
``outputs/optuna/journal.log``, che viveva sulla vecchia macchina e non e' mai
stato in git. Senza journal non puo' funzionare. Qui gli iperparametri sono
scritti esplicitamente, ripresi uno per uno dalle config delle run originali
registrate su W&B: sono quindi verificabili leggendo questo file, senza
dipendere da uno stato esterno che si puo' perdere di nuovo.

Il protocollo e' unico per tutti i bracci (vedi ``PROTOCOL``): un solo ambiente
condiviso fra training e raccolta del feedback, un solo worker, un solo reward
model. ``train_freq`` sale a 16 perche' con ``n_envs=1`` il rapporto fra passi
di gradiente e transizioni raccolte resterebbe altrimenti il doppio di quello
su cui gli iperparametri sono stati tarati:

    replay ratio = gradient_steps / (train_freq * n_envs) = 32 / (16 * 1) = 2.0

I core 0-15 sono riservati ad altri utenti: si parte da CORE_FIRST.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_hybrid_sac.py"
# Etichetta della campagna. Entra nel nome del gruppo, quindi le run di due
# campagne convivono nello stesso progetto W&B senza mescolarsi: "v2" e' quella
# con lo schedule delle query corretto (le query non si ammucchiano piu' in
# fondo alla corsa). Il parser dei gruppi legge th_<tag>_B<livello> e il tag
# puo' contenere underscore, quindi th_v2_hybrid_soft_B10 resta leggibile.
CAMPAIGN = "v2"

OUTPUT_ROOT = REPO_ROOT / "outputs" / "thesis_runs"
LOG_ROOT = OUTPUT_ROOT / "logs"

WANDB_ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "thesis-final"

# I core 0-15 non si toccano.
CORE_FIRST, CORE_LAST = 16, 63

BUDGETS = (10, 100, 1000)
SEEDS = (1, 2, 3)

# Numero di confronti, quando va scollegato dal budget delle dimostrazioni.
# None = total_queries e' il budget, che e' il protocollo normale. Un intero lo
# fissa: serve a far girare un ibrido sotto la soglia di alpha, tenendo B
# dimostrazioni. La leggono sia arm_overrides sia validate, quindi non possono
# divergere.
QUERIES_OVERRIDE = None

# --- protocollo comune a tutti i bracci -------------------------------------
# Cambiare una di queste righe cambia TUTTI i bracci insieme: e' il punto.
PROTOCOL = (
    "env.kwargs.ego=continuous",
    "env.kwargs.reward=fast",
    "env.n_envs=1",
    "env.shared_rollout_env=true",
    "agent.kwargs.device=cpu",
    "agent.kwargs.train_freq=16",          # replay ratio 2.0 con n_envs=1
    "agent.kwargs.gradient_steps=32",
    "agent.kwargs.learning_rate=0.0001242983309370202",
    "agent.kwargs.buffer_size=300000",
    "agent.kwargs.learning_starts=2000",
    "agent.kwargs.batch_size=256",
    "agent.kwargs.gamma=0.995",
    "agent.kwargs.tau=0.005",
    "agent.kwargs.ent_coef=auto",
    "agent.kwargs.policy_kwargs.net_arch=[64,64]",
    "algo.kwargs.demo_mode=gcl",
    "algo.kwargs.loss_type=demo_2",
    "algo.kwargs.relabel_rewards=true",
    "algo.kwargs.preference_fragment_length=1",
    "algo.kwargs.exploration_frac=0.0",
    "algo.kwargs.query_schedule=constant",
    "algo.kwargs.fragmenter_type=random",
    "algo.kwargs.agent_log_timestep_interval=10000",
    # n_ensembles=3: col bootstrap che torna legittimo (tre membri da
    # decorrelare) e la ricompensa vista dall'agente che e' la media di tre
    # reti, quindi un bersaglio meno mobile per SAC.
    "algo.kwargs.reward_model_kwargs.n_ensembles=3",
    "algo.kwargs.reward_model_kwargs.activation_fn=tanh",
    "algo.kwargs.batch_size_model=64",
    "train.kwargs.total_timesteps=2000000",
    "train.kwargs.timesteps_per_iteration=20000",
    "train.kwargs.log_interval=100",
    "train.kwargs.checkpoint_interval=1000000",
    "train.kwargs.scatter_interval=0",
    "eval.n_episodes=20",
)

# La normalizzazione del reward vista dall'agente e' per braccio, non globale:
# ogni metodo deve girare nella configurazione in cui e' stato tarato, altrimenti
# il confronto penalizza chi e' fuori dal suo punto di lavoro. I due bracci
# solo-preferenze sono stati tarati con la normalizzazione attiva, tutti gli
# altri senza. E' un'asimmetria dichiarata, da riportare in tesi.


@dataclass(frozen=True)
class Arm:
    """Un braccio del confronto.

    ``uses_pref``/``uses_demo`` dicono quali canali sono accesi; il budget B
    entra in quelli accesi e in nessun altro.
    """
    uses_pref: bool
    uses_demo: bool
    labels: str | None            # "soft" | "binary_bernoulli" | None
    fusion: str                   # alpha_norm_single_adam | norm_balance
    lr_rew: float
    l2_rew: float
    gradient_steps_rew: int
    batch_size_expert: int
    batch_size_pref: int
    net_arch: str
    initial_agent_timesteps: int
    pref_temperature: float | None = None
    label_smoothing: float = 0.0
    initial_queries_frac: float = 0.10   # quota del budget raccolta nel bootstrap
    normalize: bool = False              # trasformazione affine agent-facing


# Iperparametri presi dalle config W&B delle run originali. I due bracci
# "unweighted" sono l'ablazione della pesatura: stessa ricetta del loro
# omologo pesato, cambia SOLO la fusione. Nell'archivio l'ablazione soft
# girava con batch_size_expert=16 invece di 64; qui e' allineata a 64, cosi'
# fra i due bracci cambia una cosa sola.
#
# batch_size_pref: 64 sui bracci solo-preferenze, 256 sugli ibridi. Non e' un
# iperparametro come gli altri, perche' entra in alpha come B = min(batch, N):
# con 256 il canale preferenze risulta mediato su quattro volte piu' campioni a
# B=1000, e alpha scende a ~0.35 invece di ~0.82. Le prestazioni non cambiano
# (differenze fra -2.8 e +2.4 su quattro confronti appaiati), cambia cosa alpha
# descrive: col 256 il rumore del gradiente EFFETTIVAMENTE applicato, che e' la
# motivazione dello pseudocodice originale ed e' la configurazione delle lane
# di riferimento.
#
# initial_agent_timesteps uniforme a 20000: e' protocollo, non metodo, e
# lasciarlo per braccio dava a pref_soft e hybrid_bern l'1,9% di interazioni
# ambientali in piu' degli altri.
#
# hybrid_soft net_arch e hybrid_bern gradient_steps_rew vengono dalla taratura
# sotto prova 1 (40 trial per braccio, B=1000): sono le due sole modifiche con
# supporto -- [128,128] ha mediana 58.2 contro 53.7 su 16 e 9 trial, e
# gradient_steps_rew >=120 da' 54.7 contro 49.9. lr_rew e l2_rew restano dove
# erano: fra i primi dieci trial variano di tre ordini di grandezza a parita'
# di risultato, quindi il dato non li distingue.

ARMS: dict[str, Arm] = {
    "demo_only": Arm(
        uses_pref=False, uses_demo=True, labels=None, fusion="norm_balance",
        lr_rew=0.0009187069964354144, l2_rew=5.061862748858848e-06,
        gradient_steps_rew=100, batch_size_expert=16, batch_size_pref=128,
        net_arch="[64,64]", initial_agent_timesteps=20000,
        initial_queries_frac=0.0,
    ),
    "pref_soft": Arm(
        uses_pref=True, uses_demo=False, labels="soft", fusion="norm_balance",
        lr_rew=0.001837324265850939, l2_rew=0.00012704069184662418,
        gradient_steps_rew=23, batch_size_expert=64, batch_size_pref=64,
        net_arch="[32,32]", initial_agent_timesteps=20000,
        pref_temperature=20.0, initial_queries_frac=0.05, normalize=True,
    ),
    "pref_bern": Arm(
        uses_pref=True, uses_demo=False, labels="binary_bernoulli", fusion="norm_balance",
        lr_rew=0.0008519268053820848, l2_rew=1.1190973215409014e-06,
        gradient_steps_rew=99, batch_size_expert=64, batch_size_pref=64,
        net_arch="[128,128]", initial_agent_timesteps=20000,
        pref_temperature=3.0595414013726767, label_smoothing=0.1,
        initial_queries_frac=0.20, normalize=True,
    ),
    "hybrid_soft": Arm(
        uses_pref=True, uses_demo=True, labels="soft", fusion="alpha_norm_single_adam",
        lr_rew=0.001154295698198038, l2_rew=1.1265276323434602e-06,
        gradient_steps_rew=139, batch_size_expert=64, batch_size_pref=256,
        net_arch="[128,128]", initial_agent_timesteps=20000,
        pref_temperature=20.0,
    ),
    "hybrid_bern": Arm(
        uses_pref=True, uses_demo=True, labels="binary_bernoulli",
        fusion="alpha_norm_single_adam",
        lr_rew=0.0003080841576274553, l2_rew=0.0005307422191330497,
        gradient_steps_rew=145, batch_size_expert=64, batch_size_pref=256,
        net_arch="[32,32]", initial_agent_timesteps=20000,
        pref_temperature=3.0595414013726767, label_smoothing=0.1,
    ),
    "unw_soft": Arm(
        uses_pref=True, uses_demo=True, labels="soft", fusion="norm_balance",
        lr_rew=0.001154295698198038, l2_rew=1.1265276323434602e-06,
        gradient_steps_rew=139, batch_size_expert=64, batch_size_pref=256,
        net_arch="[128,128]", initial_agent_timesteps=20000,
        pref_temperature=20.0,
    ),
    "unw_bern": Arm(
        uses_pref=True, uses_demo=True, labels="binary_bernoulli", fusion="norm_balance",
        lr_rew=0.0003080841576274553, l2_rew=0.0005307422191330497,
        gradient_steps_rew=145, batch_size_expert=64, batch_size_pref=256,
        net_arch="[32,32]", initial_agent_timesteps=20000,
        pref_temperature=3.0595414013726767, label_smoothing=0.1,
    ),
}


@dataclass(frozen=True)
class Task:
    arm: str
    budget: int
    seed: int

    @property
    def group(self) -> str:
        prefix = f"th_{CAMPAIGN}" if CAMPAIGN else "th"
        return f"{prefix}_{self.arm}_B{self.budget}"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.group / self.run_name

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.run_name}.log"


def initial_queries(arm: Arm, budget: int) -> int:
    """Quota del budget raccolta prima che parta lo schedule regolare.

    Il minimo di 1 quando il braccio usa preferenze: con 0 il bootstrap
    resterebbe senza feedback e la prima iterazione senza segnale.
    """
    if not arm.uses_pref:
        return 0
    return max(1, round(arm.initial_queries_frac * budget))


def total_queries(arm: Arm, budget: int) -> int:
    """Confronti concessi al braccio: il budget, salvo override di campagna."""
    if not arm.uses_pref:
        return 0
    return budget if QUERIES_OVERRIDE is None else int(QUERIES_OVERRIDE)


def arm_overrides(name: str, budget: int, seed: int) -> list[str]:
    """Tutti gli override Hydra di una run, protocollo compreso."""
    arm = ARMS[name]
    task = Task(name, budget, seed)
    ov = [
        *PROTOCOL,
        f"algo.kwargs.lr_rew={arm.lr_rew}",
        f"algo.kwargs.l2_rew={arm.l2_rew}",
        f"algo.kwargs.gradient_steps_rew={arm.gradient_steps_rew}",
        f"algo.kwargs.batch_size_expert={arm.batch_size_expert}",
        f"algo.kwargs.batch_size_pref={arm.batch_size_pref}",
        f"algo.kwargs.reward_model_kwargs.net_arch={arm.net_arch}",
        f"algo.kwargs.initial_agent_timesteps={arm.initial_agent_timesteps}",
        f"algo.kwargs.gcl_fusion={arm.fusion}",
        f"algo.kwargs.label_smoothing={arm.label_smoothing}",
        f"algo.kwargs.normalize_agent_reward={str(arm.normalize).lower()}",
        # Canale preferenze: acceso solo se il braccio lo usa.
        f"algo.kwargs.total_queries={total_queries(arm, budget)}",
        f"algo.kwargs.initial_queries={initial_queries(arm, budget)}",
        # Canale dimostrazioni: demo_weight=0 lo spegne del tutto.
        f"algo.kwargs.demo_weight={1.0 if arm.uses_demo else 0.0}",
        f"run.seed={seed}",
        f"run.output_dir={task.output_dir.relative_to(REPO_ROOT)}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"wandb.entity={WANDB_ENTITY}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.tags=[thesis_final,{CAMPAIGN},{name},B{budget}]",
    ]
    if arm.uses_pref:
        ov.append(f"algo.kwargs.labels_type={arm.labels}")
        ov.append(f"algo.kwargs.pref_temperature={arm.pref_temperature}")
    if arm.uses_demo:
        # Il sottocampione delle dimostrazioni e' annidato e indipendente dal
        # seed di training: a parita' di budget ogni braccio vede le stesse.
        ov.append(f"run.n_expert_trajectories={budget}")
    return ov


def build_command(name: str, budget: int, seed: int, core: int) -> list[str]:
    return ["taskset", "-c", str(core), sys.executable, str(TRAIN_SCRIPT),
            *arm_overrides(name, budget, seed)]


# --- validazione ------------------------------------------------------------

def validate(name: str, budget: int, seed: int) -> dict:
    """Risolve la config con Hydra e controlla cosa atterra davvero.

    Intercetta un override sbagliato prima di ore di calcolo, non dopo. In
    particolare tiene d'occhio le chiavi che un errore renderebbe invisibile:
    l'ambiente condiviso, il numero di worker, il replay ratio e la fusione.
    """
    from omegaconf import OmegaConf

    arm = ARMS[name]
    cmd = build_command(name, budget, seed, CORE_FIRST)
    # via "taskset -c <core>"; i flag di Hydra vanno DOPO il path dello script
    cmd = cmd[3:5] + ["--cfg", "job", "--resolve"] + cmd[5:]
    res = subprocess.run(cmd, cwd=REPO_ROOT, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"[{name} B={budget}] Hydra non risolve la config:\n"
                           + res.stderr.strip()[-1500:])
    cfg = OmegaConf.create(res.stdout)
    # Le chiavi di protocollo si leggono da PROTOCOL invece di riscriverle qui:
    # erano due elenchi della stessa cosa, e quando ho portato n_ensembles a 3
    # ne ho aggiornato uno solo e il lancio si e' fermato.
    def _da_protocollo(chiave, conv):
        for o in PROTOCOL:
            k, v = o.split("=", 1)
            if k == chiave:
                return conv(v)
        raise KeyError(chiave)

    attesi = {
        "env.n_envs": _da_protocollo("env.n_envs", int),
        "env.shared_rollout_env": _da_protocollo(
            "env.shared_rollout_env", lambda v: v.lower() == "true"),
        "agent.kwargs.train_freq": _da_protocollo("agent.kwargs.train_freq", int),
        "agent.kwargs.gradient_steps": _da_protocollo(
            "agent.kwargs.gradient_steps", int),
        "algo.kwargs.gcl_fusion": arm.fusion,
        "algo.kwargs.loss_type": "demo_2",
        "algo.kwargs.reward_model_kwargs.n_ensembles": _da_protocollo(
            "algo.kwargs.reward_model_kwargs.n_ensembles", int),
        "algo.kwargs.normalize_agent_reward": arm.normalize,
        "algo.kwargs.gradient_steps_rew": arm.gradient_steps_rew,
        "algo.kwargs.batch_size_expert": arm.batch_size_expert,
        "algo.kwargs.batch_size_pref": arm.batch_size_pref,
        "algo.kwargs.label_smoothing": arm.label_smoothing,
        "algo.kwargs.total_queries": total_queries(arm, budget),
        "algo.kwargs.initial_queries": initial_queries(arm, budget),
        "algo.kwargs.demo_weight": 1.0 if arm.uses_demo else 0.0,
        "run.demo_subsample_seed": 1000,
        "run.seed": seed,
        "train.kwargs.total_timesteps": _da_protocollo(
            "train.kwargs.total_timesteps", int),
    }
    if arm.uses_demo:
        attesi["run.n_expert_trajectories"] = budget
    if arm.uses_pref:
        attesi["algo.kwargs.labels_type"] = arm.labels
        attesi["algo.kwargs.pref_temperature"] = arm.pref_temperature
    # Il bootstrap non sta nel PROTOCOL di serie (lo decide n_ensembles), ma se
    # una campagna lo fissa esplicitamente va controllato come le altre chiavi:
    # sbagliarlo cambia quanti confronti distinti vede il reward model.
    try:
        attesi["algo.kwargs.bootstrap_comparisons"] = _da_protocollo(
            "algo.kwargs.bootstrap_comparisons",
            lambda v: None if v.lower() in ("null", "none") else v.lower() == "true")
    except KeyError:
        pass
    sbagliati = {}
    for key, atteso in attesi.items():
        got = OmegaConf.select(cfg, key)
        if got != atteso:
            sbagliati[key] = (got, atteso)
    if sbagliati:
        raise RuntimeError(f"[{name} B={budget}] config risolta diversa: {sbagliati}")
    # Il replay ratio e' il motivo per cui train_freq non e' 8.
    ratio = cfg.agent.kwargs.gradient_steps / (cfg.agent.kwargs.train_freq * cfg.env.n_envs)
    if abs(ratio - 2.0) > 1e-9:
        raise RuntimeError(f"[{name}] replay ratio {ratio}, atteso 2.0")
    return {"arm": name, "budget": budget, "seed": seed, "replay_ratio": ratio}


# --- lancio -----------------------------------------------------------------

def preflight() -> None:
    """Controlli che costano un secondo e valgono ore di calcolo.

    La validazione Hydra dice che la config e' giusta, non che la run partira':
    senza login W&B ogni processo muore dentro ``wandb.init`` dopo essere stato
    avviato, e quarantacinque processi falliscono uno per uno senza che nessuno
    se ne accorga finche' non si guardano i log.
    """
    import os

    if os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled", "dryrun"):
        print(f"  W&B in modalita' {os.environ['WANDB_MODE']}: niente login richiesto")
        return
    import wandb

    if wandb.api.api_key is None:
        raise SystemExit(
            "W&B non ha credenziali su questa macchina: ogni run morirebbe in\n"
            "wandb.init dopo essere stata avviata. Scegli una via:\n"
            "  1) autenticati una volta:  wandb login\n"
            "  2) lancia offline e sincronizza dopo:  WANDB_MODE=offline ...\n"
            "     poi  wandb sync outputs/thesis_runs/*/*/wandb/offline-run-*"
        )
    print(f"  W&B ok (entity {WANDB_ENTITY}, project {WANDB_PROJECT})")


def busy_cores() -> set[int]:
    """Core gia' occupati da un training, letti dall'affinita' reale."""
    busy = set()
    out = subprocess.run(["ps", "-eo", "pid,args"], text=True,
                         stdout=subprocess.PIPE).stdout.splitlines()
    for line in out:
        if "train_hybrid_sac.py" not in line or "ps -eo" in line:
            continue
        pid = line.split()[0]
        aff = subprocess.run(["taskset", "-pc", pid], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        for chunk in aff.rsplit(":", 1)[-1].strip().split(","):
            if "-" in chunk:
                a, b = chunk.split("-")
                busy.update(range(int(a), int(b) + 1))
            elif chunk.isdigit():
                busy.add(int(chunk))
    return busy


def launch(tasks: list[Task]) -> list[dict]:
    # Dal 63 a scendere: i core bassi sono i piu' contesi dagli altri utenti e
    # lo 0-15 e' comunque vietato, quindi si parte dal fondo.
    busy = busy_cores()
    free = [c for c in range(CORE_LAST, CORE_FIRST - 1, -1) if c not in busy]
    if len(tasks) > len(free):
        raise RuntimeError(
            f"{len(tasks)} run ma solo {len(free)} core liberi fra {CORE_FIRST} e "
            f"{CORE_LAST}. Lancia meno bracci per volta, o aspetta."
        )
    launched = []
    for task, core in zip(tasks, free):
        task.output_dir.mkdir(parents=True, exist_ok=True)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_command(task.arm, task.budget, task.seed, core)
        with open(task.log_path, "w") as log:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        launched.append({"run_name": task.run_name, "arm": task.arm,
                         "budget": task.budget, "seed": task.seed,
                         "pid": proc.pid, "core": core,
                         "log": str(task.log_path.relative_to(REPO_ROOT))})
        print(f"lanciata {task.run_name} pid={proc.pid} core={core}")
    return launched


def status() -> int:
    path = OUTPUT_ROOT / "manifest.json"
    if not path.exists():
        print("Nessun manifest: non e' stato lanciato niente.")
        return 1
    manifest = json.loads(path.read_text())
    print(f"{'run':<34} {'pid':>8}  {'stato':<9} {'iter':>5}")
    print("-" * 62)
    for e in manifest["runs"]:
        # train_hybrid_sac.py crea una sua sottocartella col nome della run
        # DENTRO run.output_dir, quindi il livello e' doppio.
        task = Task(e["arm"], e["budget"], e["seed"])
        run_dir = task.output_dir / e["run_name"]
        done = (run_dir / "final_eval.json").exists()
        alive = Path(f"/proc/{e['pid']}").exists()
        stato = "finita" if done else ("in corso" if alive else "FERMA")
        metrics = run_dir / "metrics.jsonl"
        n = sum(1 for _ in metrics.open()) if metrics.exists() else 0
        print(f"{e['run_name']:<34} {e['pid']:>8}  {stato:<9} {n:>5}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=["all"],
                    help=f"bracci da lanciare, o 'all' ({', '.join(ARMS)})")
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--dry-run", action="store_true",
                    help="valida le config e stampa un comando, senza lanciare")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        return status()

    names = list(ARMS) if args.arms == ["all"] else args.arms
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"braccio sconosciuto: {n!r}. Disponibili: {', '.join(ARMS)}")

    tasks = [Task(n, b, s) for n in names for b in args.budgets for s in args.seeds]
    print(f"{len(tasks)} run: {len(names)} bracci x {len(args.budgets)} budget "
          f"x {len(args.seeds)} seed")

    for n in names:
        for b in args.budgets:
            rec = validate(n, b, args.seeds[0])
            print(f"  ok {n:<12} B={b:<5} replay_ratio={rec['replay_ratio']:.1f}")

    if args.dry_run:
        print("\n" + shlex.join(build_command(names[0], args.budgets[0], args.seeds[0], CORE_FIRST)))
        return 0

    preflight()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    launched = launch(tasks)
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps({
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wandb_project": WANDB_PROJECT,
        "normalize_agent_reward": {n: ARMS[n].normalize for n in ARMS},
        "runs": launched,
    }, indent=2))
    print(f"\nmanifest: {(OUTPUT_ROOT / 'manifest.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
