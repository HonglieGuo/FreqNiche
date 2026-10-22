"""Audit-only entry that restores console streams after the frozen runner closes its log."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit_arguments(arguments):
    arguments = list(arguments)
    if "--phase" in arguments:
        i = arguments.index("--phase")
        if i + 1 >= len(arguments) or arguments[i + 1] != "audit":
            raise ValueError("This entry allows only --phase audit")
    else:
        arguments += ["--phase", "audit"]
    if any(a.startswith("--phase=") and a != "--phase=audit" for a in arguments):
        raise ValueError("This entry allows only --phase audit")
    if "--replay" not in arguments:
        arguments.append("--replay")
    return arguments


def main():
    arguments = audit_arguments(sys.argv[1:])
    from experiments_analysis.ablation import run_cosmx_all_three_ablation as run
    previous_argv = sys.argv
    console = sys.stdout, sys.stderr
    try:
        sys.argv = [run.__file__, *arguments]
        run.main()
    finally:
        # Do not suppress audit exceptions; only restore streams before interpreter shutdown.
        sys.stdout, sys.stderr = console
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
