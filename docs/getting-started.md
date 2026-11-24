# Getting started

Sparksi is the Peelsa model manager for NVIDIA DGX Spark. One controller process (`sparksi serve`) provides the dashboard and HTTP API on port 7500 by default. Each Spark runs a small announcer (`sparksi-announce`) that sends one UDP datagram every 3 seconds to the controller's announce port (default 7501). Node operations go over SSH in BatchMode; the node agent is sent over SSH per operation and is never installed.

## Requirements

- Python 3.12 or newer on the controller. The only runtime dependency is PyYAML. The Sparks need only the system `python3` for the agent and announcer.
- One to four NVIDIA DGX Spark machines.
- Docker on each Spark, for container engines.
- Key-based SSH from the controller to each Spark. `ssh -o BatchMode=yes <spark> true` must succeed without a prompt.
- A tailnet (for example Tailscale) between controller and Sparks. The dashboard listens on loopback plus one tailnet address. `0.0.0.0`, `::` and `*` are always refused; see [Listen addresses](config.md#listen-addresses) for the deny lists.

The controller can run on the head Spark. In that case give that Spark `ssh = "local"` in the configuration. It can also run on another machine on the tailnet.

## Install

From a clone:

```sh
git clone <repo-url> sparksi
cd sparksi
python3 -m venv .venv
.venv/bin/pip install -e .
```

Extras:

- `.venv/bin/pip install -e '.[dev]'` adds pytest and ruff for development.
- `.[monitor]` adds psutil for the terminal monitor.

Optional: `scripts/install-cli.sh` writes a `~/bin/sparksi` launcher. It does not touch systemd or models.

## Configure

```sh
mkdir -p ~/.config/sparksi
cp config/sparksi.example.toml ~/.config/sparksi/sparksi.toml
cp config/catalog.example.yaml ~/.config/sparksi/catalog.yaml
```

Edit both files.

- `sparksi.toml` holds every host, address, path and user name.
- `catalog.yaml` lists the models (library rows) and services.

Neither file is committed.

Config search order:

1. `$SPARKSI_CONFIG`
2. `$XDG_CONFIG_HOME/sparksi/sparksi.toml` (default `~/.config/sparksi/sparksi.toml`)
3. `./config/sparksi.toml`

See [config.md](config.md) for one to four Sparks and every key.

## First run

```sh
.venv/bin/sparksi --version
.venv/bin/sparksi status        # read-only probe of what is resident
.venv/bin/sparksi serve         # dashboard on http://127.0.0.1:7500
```

`sparksi serve --port 7510` runs a second copy on another port for a dry check.

Open `http://127.0.0.1:7500` (or the controller's tailnet address on port
7500) in a browser. Overview shows one card per configured Spark, including the live GPU clock speed.

The lock is ON when Sparksi starts with no lock file. You can look at
everything, but Load, Unload and Network edits wait until you press Unlock.
See [lock.md](lock.md). A GPU max-clock cap is set on Options, does not need Unlock, and is not kept across a reboot.

To load a model:

1. Open Model Manager and pick a model from the list.
2. Choose a placement (one Spark, or two Sparks for a model that does not fit
   one) and a port between 8010 and 8099.
3. Press Dry run. The plan is READY, or REFUSE with every reason.
4. Press Unlock, then Load, and confirm the model, machines and port.

Save the settings as a recipe so the next load is one click. See
[recipes.md](recipes.md).

### Announcer on each Spark

The dashboard learns what each Spark is running from its announcer. From the
controller, copy the announcer to each Spark over SSH:

```sh
scripts/install-announce.sh --host <spark>
```

This copies one file to `~/.local/bin/sparksi-announce` on that Spark. It does
not restart or enable anything unless you pass `--restart`.

Then print a user systemd unit for that Spark and install it there yourself:

```sh
.venv/bin/sparksi install-unit --print --unit announce --node-id <id> --controller <controller-tailnet-ip>:7501
```

`--node-id` is the `[[nodes]]` id from `sparksi.toml`, not the host name.

Allow UDP 7501 on the controller only from the Sparks' tailnet addresses.

## Run as a user service

Print a user unit for the controller, review it, then enable it:

```sh
mkdir -p ~/.config/systemd/user
.venv/bin/sparksi install-unit --print --unit controller --prefix "$PWD" \
  > ~/.config/systemd/user/sparksi-controller.service
"${EDITOR:-vi}" ~/.config/systemd/user/sparksi-controller.service   # review it
systemctl --user daemon-reload
systemctl --user enable --now sparksi-controller.service
```

`install-unit` only prints. It never enables or starts anything.

Full steps, including the firewall and SSH checks, are in [deploy.md](deploy.md).

## Next steps

- [config.md](config.md): `sparksi.toml`, one to four Sparks
- [pages.md](pages.md): the dashboard pages
- [recipes.md](recipes.md): saved recipes, naming and the default
- [lock.md](lock.md): the lock and the confirmations
- [cli.md](cli.md): command reference
- [deploy.md](deploy.md): running under systemd