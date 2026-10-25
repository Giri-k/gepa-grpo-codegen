"""Run a few small programs in the sandbox and print what happens.

    uv run python scripts/sandbox_demo.py

Needs Docker Desktop running. Each program gets the same two test inputs.
"""

from codegen.sandbox import Limits, Sandbox

INPUTS = ["3\n", "10\n"]  # each string is fed to the program's stdin, one run per string

PROGRAMS = {
    "1. correct program (doubles n)": "n = int(input())\nprint(n * 2)",
    "2. crashes (division by zero)": "n = int(input())\nprint(n / 0)",
    "3. infinite loop": "while True:\n    pass",
    "4. syntax error": "print(",
    "5. tries to read your Mac's files": "import os\nprint(os.listdir('/Users'))",
    "6. tries to use the internet": (
        "import socket\n"
        "socket.create_connection(('google.com', 80), timeout=2)\n"
        "print('connected!')"
    ),
    "7. shows where it is running": (
        "import os, platform\n"
        "print('machine:', platform.system(), platform.machine())\n"
        "print('folder: ', os.getcwd())\n"
        "print('files at /:', sorted(os.listdir('/')))"
    ),
}


def main() -> None:
    print("Starting 1 sandbox container...")
    with Sandbox(workers=1, limits=Limits(timeout=2.0)) as sb:
        for title, code in PROGRAMS.items():
            print("\n" + "=" * 70)
            print(title)
            print("-" * 70)
            print(code)
            print("-" * 70)
            result = sb.run(code, INPUTS)
            if result.status != "ok":
                print(f"status: {result.status}  ->  {result.error}")
                continue
            for inp, t in zip(INPUTS, result.tests):
                print(f"input {inp.strip()!r:>6}  outcome: {t.outcome:<14} time: {t.seconds:.2f}s")
                if t.stdout:
                    print("   printed:", t.stdout.rstrip().replace("\n", "\n            "))
                if t.stderr:
                    print("   error:  ", t.stderr.strip().splitlines()[-1])
    print("\nDone. Container stopped and removed.")


if __name__ == "__main__":
    main()
