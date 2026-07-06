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
from typing import Any, cast

_parser_cls = cast(Any, argparse.ArgumentParser)

_original_check_help = _parser_cls._check_help


def patch_argparse_check_help() -> None:
    if _parser_cls._check_help is not _original_check_help:
        return

    def _check_help(self, action):
        try:
            _original_check_help(self, action)
        except TypeError:
            return

    _parser_cls._check_help = _check_help
