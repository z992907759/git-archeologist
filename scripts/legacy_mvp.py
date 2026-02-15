# Socratic Git CLI compatibility entrypoint.
# Preferred entry: python -m socratic_git.cli ...
# Backward-compatible entry: python socratic_mvp.py ...

from socratic_git.cli import main


if __name__ == "__main__":
    main()
