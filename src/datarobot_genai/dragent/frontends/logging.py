# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


def logging_handler_setup() -> None:
    # Noisy NAT messages that are safe to suppress globally.
    suppressed_nat_messages = [
        "StepAdaptor is disabled",
        "Dask is not installed",
        "Dask is not available",
        "feature is experimental and the API may change",
        "Using provided input_schema for multi-argument function",
    ]

    orig_handler_handle = logging.Handler.handle

    def _filtered_handle(self: logging.Handler, record: logging.LogRecord) -> bool | None:
        try:
            msg = record.getMessage()
        except Exception:
            return orig_handler_handle(self, record)

        # Drop log records that match any of the suppressed NAT messages.
        if any(s in msg for s in suppressed_nat_messages):
            return None

        # Strip newlines from the format string so multi-line messages are
        # collapsed to a single line. This keeps log output clean in environments
        # that treat each line as a separate log entry (e.g. structured logging).
        if isinstance(record.msg, str):
            record.msg = record.msg.replace("\n", " ")

        # Also strip newlines from any string values in the format args so that
        # newlines coming from interpolated values are removed too.
        # The structure and length of record.args is preserved so downstream
        # handlers that unpack it (e.g. structured/JSON loggers) continue to work.
        if isinstance(record.args, dict):
            record.args = {
                k: v.replace("\n", " ") if isinstance(v, str) else v for k, v in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(
                v.replace("\n", " ") if isinstance(v, str) else v for v in record.args
            )

        return orig_handler_handle(self, record)

    logging.Handler.handle = _filtered_handle  # type: ignore[assignment]
