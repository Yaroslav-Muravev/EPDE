"""Data adapter for Navier-Stokes (placeholder).

NS is excluded from smoke runs. Truth equations + loader will be filled
in once the Re-specific ground-truth pair + continuity equation are
pinned down (mirror ``ns.py:ns_data`` on cylinder_nektar_wake.mat).
"""


def load_data():
    raise NotImplementedError(
        'NS data loader not implemented yet; mirror ns.py:ns_data on '
        'cylinder_nektar_wake.mat once truth tokens are pinned.'
    )
