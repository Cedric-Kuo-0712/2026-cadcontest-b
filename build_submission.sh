#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/docker/build_submission.sh" "$@"
