#!/usr/bin/env python3
"""Tiny fake engine used only to exercise GTP framing and buffering."""

import sys
import time
import os


sys.stdout.write("startup banner before protocol\n")
sys.stdout.flush()
for line in sys.stdin:
    command_id, command = line.strip().split(" ", 1)
    if command == "name":
        sys.stdout.write(f"={command_id} Fake")
        sys.stdout.flush()
        time.sleep(0.01)
        sys.stdout.write("Hex\n\n")
        sys.stdout.flush()
    elif command == "two":
        # Two frames in one write verifies that trailing output is retained.
        sys.stdout.write(f"={command_id} first\n\n={int(command_id) + 1} trailing\n\n")
        sys.stdout.flush()
    elif command == "prebuffered":
        # The response was deliberately emitted with the preceding command.
        continue
    elif command == "hang":
        # No response exercises client-side timeout handling without sleeping.
        continue
    elif command == "error":
        sys.stdout.write(f"?{command_id} deliberate error\n\n")
        sys.stdout.flush()
    elif command == "crash":
        sys.stderr.write("deliberate fake startup/process crash\n")
        sys.stderr.flush()
        os._exit(23)
    elif command == "quit":
        sys.stdout.write(f"={command_id}\n\n")
        sys.stdout.flush()
        break
    else:
        sys.stdout.write(f"={command_id}\n\n")
        sys.stdout.flush()
