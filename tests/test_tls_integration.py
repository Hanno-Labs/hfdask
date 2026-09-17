import pytest
from distributed import Client, LocalCluster

from hfdask.network import issue_credentials


def test_real_dask_mtls_accepts_cluster_client_and_rejects_another_ca():
    server, client = issue_credentials(["server"], client_name="client")
    outsider, _ = issue_credentials(["outsider"])
    assert client is not None

    with server[0].security() as server_security:
        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            host="127.0.0.1",
            protocol="tls",
            security=server_security,
            dashboard_address=None,
        ) as cluster:
            with client.security() as client_security:
                with Client(
                    cluster.scheduler_address,
                    security=client_security,
                    set_as_default=False,
                ) as connected:
                    assert connected.submit(abs, -41).result(timeout=10) == 41

            with outsider[0].security() as outsider_security:
                with pytest.raises(OSError):
                    Client(
                        cluster.scheduler_address,
                        security=outsider_security,
                        timeout=1,
                        set_as_default=False,
                    )
