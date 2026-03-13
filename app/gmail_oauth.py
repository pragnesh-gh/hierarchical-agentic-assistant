"""One-time OAuth helper for Gmail API token generation."""

from tools_email import _load_credentials


def main() -> None:
    _load_credentials(interactive=True)
    print("OAuth token saved to secrets/token.json")


if __name__ == "__main__":
    main()
