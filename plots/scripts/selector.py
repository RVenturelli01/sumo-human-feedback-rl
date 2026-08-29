#!/usr/bin/env python
"""Interactive run selector: filters, coverage and a live preview.

    python plots/scripts/selector.py            # http://127.0.0.1:8770
    python plots/scripts/selector.py --port 9000

It serves a local page with no external dependencies. The index is reread on
every query, so runs added by build_index.py appear without a restart.

"Save selection" writes the whole figure -- filters, grid, colours -- so that
`plot_curves.py --runs-file` or `plot_budget.py --runs-file` redraw exactly it.

The code is in rtplots/webui/: `api.py` the logic, `server.py` the transport.
"""
import argparse

import _bootstrap  # noqa: F401
from rtplots.webui import serve


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default, usa il port forwarding di VSCode) o 0.0.0.0")
    args = p.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
