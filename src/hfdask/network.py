"""Hugging Face network-group addressing and per-cluster Dask mTLS credentials."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from distributed.security import Security

SCHEDULER_ALIAS = "scheduler"
SCHEDULER_PORT = 8786
WORKER_PORT_BASE = 10_000
NANNY_PORT_BASE = 20_000
_MAX_ORDINAL = 9_999
_GROUP = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?\Z")
_ALIAS = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?\Z")
_SECRET_NAMES = ("HFDASK_TLS_CA", "HFDASK_TLS_CERT", "HFDASK_TLS_KEY")


def network_group(cluster_id: str) -> str:
    """Return a valid, unique HF network-group name for a cluster handle."""
    value = f"hfdask-{cluster_id}"
    if not _GROUP.fullmatch(value):
        raise ValueError("Cluster ID cannot form a valid HF network-group name")
    return value


def node_alias(node: int) -> str:
    """Return the stable HF network alias claimed by one cluster Job."""
    value = f"node-{node}"
    if node < 0 or not _ALIAS.fullmatch(value):
        raise ValueError("Node index cannot form a valid HF network alias")
    return value


def network_hostname(alias: str, environment: Mapping[str, str] | None = None) -> str:
    """Resolve an alias to the hostname convention injected by HF Jobs."""
    if not _ALIAS.fullmatch(alias):
        raise ValueError("Invalid HF network alias")
    values = os.environ if environment is None else environment
    try:
        prefix = values["HF_NETWORK_GROUP_PREFIX"]
    except KeyError as error:
        raise RuntimeError(
            "HF_NETWORK_GROUP_PREFIX is unavailable outside a network group"
        ) from error
    if not prefix:
        raise RuntimeError("HF_NETWORK_GROUP_PREFIX must be nonempty")
    return f"{prefix}{alias}"


def scheduler_address(environment: Mapping[str, str] | None = None) -> str:
    """Return the network-group address used by every Dask worker."""
    return f"tls://{network_hostname(SCHEDULER_ALIAS, environment)}:{SCHEDULER_PORT}"


def worker_port(ordinal: int) -> int:
    if not 0 <= ordinal <= _MAX_ORDINAL:
        raise ValueError("Worker ordinal is outside the reserved port range")
    return WORKER_PORT_BASE + ordinal


def nanny_port(ordinal: int) -> int:
    if not 0 <= ordinal <= _MAX_ORDINAL:
        raise ValueError("Worker ordinal is outside the reserved port range")
    return NANNY_PORT_BASE + ordinal


@dataclass(frozen=True)
class TLSCredentials:
    """One CA-trusted Dask identity, kept out of manifests and representations."""

    ca_certificate: str = field(repr=False)
    certificate: str = field(repr=False)
    private_key: str = field(repr=False)

    def __post_init__(self) -> None:
        ca = x509.load_pem_x509_certificate(self.ca_certificate.encode())
        certificate = x509.load_pem_x509_certificate(self.certificate.encode())
        key = serialization.load_pem_private_key(self.private_key.encode(), password=None)
        if not ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
            raise ValueError("TLS CA certificate is not a certificate authority")
        if certificate.issuer != ca.subject:
            raise ValueError("TLS certificate was not issued by the supplied CA")
        certificate.verify_directly_issued_by(ca)
        public_format = {
            "encoding": serialization.Encoding.DER,
            "format": serialization.PublicFormat.SubjectPublicKeyInfo,
        }
        if certificate.public_key().public_bytes(**public_format) != key.public_key().public_bytes(
            **public_format
        ):
            raise ValueError("TLS private key does not match its certificate")

    def job_secrets(self) -> dict[str, str]:
        """Encode PEM values safely for HF Job secret environment variables."""
        values = (self.ca_certificate, self.certificate, self.private_key)
        return {
            name: base64.b64encode(value.encode()).decode()
            for name, value in zip(_SECRET_NAMES, values, strict=True)
        }

    def save(self, path: Path) -> None:
        """Create a mode-0600 JSON credential file for later persistent connections."""
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(
                    {
                        "ca_certificate": self.ca_certificate,
                        "certificate": self.certificate,
                        "private_key": self.private_key,
                    },
                    output,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> TLSCredentials:
        """Load a client identity only when no group or other permissions expose it."""
        if path.stat().st_mode & 0o077:
            raise PermissionError("TLS credential file must not be accessible by group or others")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("TLS credential file must contain a JSON object")
        try:
            return cls(
                payload["ca_certificate"],
                payload["certificate"],
                payload["private_key"],
            )
        except KeyError as error:
            raise ValueError("TLS credential file is incomplete") from error

    @classmethod
    def from_environment(
        cls, environment: MutableMapping[str, str] | None = None
    ) -> TLSCredentials:
        """Consume a Job's encoded credentials so child processes only receive file paths."""
        values = os.environ if environment is None else environment
        decoded = []
        for name in _SECRET_NAMES:
            try:
                encoded = values.pop(name)
            except KeyError as error:
                raise RuntimeError(f"Missing Job secret: {name}") from error
            try:
                decoded.append(base64.b64decode(encoded, validate=True).decode())
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid Job secret: {name}") from error
        return cls(*decoded)

    @contextmanager
    def security(self) -> Iterator[Security]:
        """Materialize private files for Dask, then remove them on context exit."""
        with tempfile.TemporaryDirectory(prefix="hfdask-tls-") as directory:
            root = Path(directory)
            root.chmod(0o700)
            ca_path = root / "ca.pem"
            cert_path = root / "identity.pem"
            key_path = root / "identity.key"
            ca_path.write_text(self.ca_certificate)
            cert_path.write_text(self.certificate)
            key_path.write_text(self.private_key)
            ca_path.chmod(0o600)
            cert_path.chmod(0o600)
            key_path.chmod(0o600)
            yield Security(
                require_encryption=True,
                tls_ca_file=str(ca_path),
                tls_client_cert=str(cert_path),
                tls_client_key=str(key_path),
                tls_scheduler_cert=str(cert_path),
                tls_scheduler_key=str(key_path),
                tls_worker_cert=str(cert_path),
                tls_worker_key=str(key_path),
            )


def issue_credentials(
    job_names: Sequence[str], *, client_name: str | None = None
) -> tuple[tuple[TLSCredentials, ...], TLSCredentials | None]:
    """Create an ephemeral cluster CA and one distinct leaf identity per principal."""
    if not job_names or len(set(job_names)) != len(job_names):
        raise ValueError("Every Job needs a distinct TLS identity")
    names = [*job_names, *([client_name] if client_name is not None else [])]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Every Job and client needs a distinct nonempty TLS identity")

    now = datetime.now(UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hfdask ephemeral CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = ca.public_bytes(serialization.Encoding.PEM).decode()

    def issue(name: str) -> TLSCredentials:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]
                ),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        return TLSCredentials(
            ca_pem,
            certificate.public_bytes(serialization.Encoding.PEM).decode(),
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode(),
        )

    issued = tuple(issue(name) for name in names)
    job_count = len(job_names)
    return issued[:job_count], issued[job_count] if client_name is not None else None
