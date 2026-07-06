"""Environment compatibility shims.

Hydra 1.3.4's internal --shell-completion argument uses a help object
(`LazyCompletionHelp`) that doesn't implement `__contains__`. Python 3.14
made argparse's `_check_help` validation stricter (it probes `'%' not in
help_string`), which now raises on that object — breaking every
`@hydra.main`-decorated script before it even parses arguments. This is a
Hydra/Python version-skew bug, not something in this project's control
(no hydra-core release fixes it yet). `_check_help` is a pure validation
step with no effect on actual parsing, so skipping it on failure is safe.
"""

import argparse

_original = getattr(argparse.ArgumentParser, "_check_help")


def _safe_check_help(self, action):
    try:
        return _original(self, action)
    except TypeError:
        # ignore Hydra LazyCompletionHelp incompatibility
        return


setattr(argparse.ArgumentParser, "_check_help", _safe_check_help)
