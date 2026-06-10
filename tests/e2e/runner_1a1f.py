#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backward-compatible 1A1F entrypoint."""

from tests.e2e.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
