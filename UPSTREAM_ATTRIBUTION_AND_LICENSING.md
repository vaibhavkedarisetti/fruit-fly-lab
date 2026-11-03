# Upstream attribution and licensing status

## Why this file exists

This local repository is a derivative working tree of the public GitHub project
[`vaibhavkedarisetti/fruit-fly-lab`](https://github.com/vaibhavkedarisetti/fruit-fly-lab).
It is not currently proven to be an independently authored codebase.

## Verified local evidence — 2026-11-03

- The original configured remote was
  `https://github.com/vaibhavkedarisetti/fruit-fly-lab.git`.
- The local branch is a descendant of upstream commit
  `26672e06427c12c61536ce1bd93dae7442944681` on `origin/main`.
- The earliest project commits in local history are authored by `vaibhav
  kedarisetti <vaibhavkedarisetti1@gmail.com>`.
- The upstream public repository has no tracked `LICENSE`, `COPYING`, or
  `NOTICE` file. Its GitHub page shows it is public, but public visibility alone
  does not grant a copyright licence to relicense, distribute, or publish a
  modified derivative.

## Consequence

Do **not** represent this code as solely authored by the current local Git user
or publish it under a new MIT licence without permission from the upstream
copyright holder. Keep upstream attribution intact.

The added local commits by Gunin Bhardwaj include validation tooling,
documentation, notebook tutorials, release preparation, and diagnostics. Those
commits do not establish ownership of the pre-existing upstream code.

## Safe options before public release

1. Ask the upstream maintainer for written permission to fork, modify, and
   publish under an agreed licence; then add that licence and attribution.
2. Keep this as a private local research/learning workspace.
3. Publish only independently authored documentation or results that do not
   redistribute upstream code or FlyWire-derived browser assets, while citing
   the upstream project and FlyWire terms.
4. Rebuild an independent implementation from the published papers and
   permitted data, with clean provenance and legal review before licensing it.

## Remote configuration

The former `origin` remote should be retained as `upstream` for attribution and
updates. A future repository owned by the current user should be added as a
new `origin` only after the licensing path above is resolved.
