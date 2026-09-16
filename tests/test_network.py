import base64
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID

from hfdask.network import (
    TLSCredentials,
    issue_credentials,
    nanny_port,
    network_group,
    network_hostname,
    node_alias,
    scheduler_address,
    worker_port,
)


def common_name(credentials):
    certificate = x509.load_pem_x509_certificate(credentials.certificate.encode())
    return certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value


def test_network_group_names_and_addresses():
    assert network_group("a" * 32) == "hfdask-" + "a" * 32
    assert node_alias(63) == "node-63"
    environment = {"HF_NETWORK_GROUP_PREFIX": "group.internal-"}
    assert network_hostname("scheduler", environment) == "group.internal-scheduler"
    assert scheduler_address(environment) == "tls://group.internal-scheduler:8786"
    assert worker_port(7) == 10007
    assert nanny_port(7) == 20007


@pytest.mark.parametrize("call", [lambda: network_group("INVALID"), lambda: node_alias(-1)])
def test_invalid_group_identity(call):
    with pytest.raises(ValueError):
        call()


def test_network_group_environment_required():
    with pytest.raises(RuntimeError, match="network group"):
        network_hostname("scheduler", {})


def test_distinct_ca_signed_credentials_and_secret_round_trip():
    jobs, client = issue_credentials(["node-0", "node-1"], client_name="client")
    assert client is not None
    assert [common_name(item) for item in (*jobs, client)] == ["node-0", "node-1", "client"]
    assert len({item.certificate for item in (*jobs, client)}) == 3
    assert len({item.private_key for item in (*jobs, client)}) == 3
    assert {item.ca_certificate for item in (*jobs, client)} == {client.ca_certificate}
    assert "PRIVATE KEY" not in repr(client)

    environment = jobs[0].job_secrets()
    assert all("BEGIN" not in value for value in environment.values())
    restored = TLSCredentials.from_environment(environment)
    assert restored == jobs[0]
    assert environment == {}


def test_job_secrets_are_base64():
    jobs, _ = issue_credentials(["node-0"])
    secrets = jobs[0].job_secrets()
    assert base64.b64decode(secrets["HFDASK_TLS_CA"]).startswith(b"-----BEGIN CERTIFICATE-----")
    assert base64.b64decode(secrets["HFDASK_TLS_KEY"]).startswith(b"-----BEGIN PRIVATE KEY-----")


def test_credentials_save_load_and_permissions(tmp_path):
    jobs, _ = issue_credentials(["node-0"])
    path = tmp_path / "client.json"
    jobs[0].save(path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert TLSCredentials.load(path) == jobs[0]
    with pytest.raises(FileExistsError):
        jobs[0].save(path)
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        TLSCredentials.load(path)


def test_security_files_are_private_and_ephemeral():
    jobs, _ = issue_credentials(["node-0"])
    paths: list[Path] = []
    with jobs[0].security() as security:
        paths = [
            Path(security.tls_ca_file),
            Path(security.tls_client_cert),
            Path(security.tls_client_key),
        ]
        assert all(path.exists() and path.stat().st_mode & 0o777 == 0o600 for path in paths)
        assert security.require_encryption is True
    assert all(not path.exists() for path in paths)


def test_mismatched_private_key_rejected():
    jobs, _ = issue_credentials(["node-0", "node-1"])
    with pytest.raises(ValueError, match="private key"):
        TLSCredentials(jobs[0].ca_certificate, jobs[0].certificate, jobs[1].private_key)
