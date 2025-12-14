import subprocess
import sys

PY = sys.executable


def run_help(cmd_args):
    proc = subprocess.run(
        [PY, "-m", "src.cli.cli_modular"] + cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def test_extract_help_has_no_url_positional():
    rc, out = run_help(["extract", "--help"])
    assert rc == 0

    # If no positional arguments are present argparse omits the header.
    pos_idx = out.find("positional arguments:")
    if pos_idx == -1:
        # No positional args -> OK
        return

    # Otherwise ensure `url` is not listed under positional args
    block = out[pos_idx : pos_idx + 400]
    assert "url" not in block


def test_extract_url_help_includes_url_positional():
    rc, out = run_help(["extract-url", "--help"])
    assert rc == 0

    pos_idx = out.find("positional arguments:")
    assert pos_idx != -1, "expected positional arguments section in extract-url help"

    block = out[pos_idx : pos_idx + 400]
    assert "url" in block


def test_extract_command_uses_batch_handler():
    """`extract` must resolve to the batch handler, not the single-URL command."""

    from src.cli import cli_modular

    parser_func, handler_func = cli_modular._load_command_parser("extract")

    assert handler_func.__name__ == "handle_extraction_command"
