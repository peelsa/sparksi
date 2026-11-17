# Deploy

These steps stay on the machine you choose to run them on. Nothing in this
repository enables or starts a service. `sparksi install-unit --print` writes
a unit to stdout. `scripts/install-announce.sh` copies one file and, unless
you pass `--restart`, does not restart anything.

Placeholders used below: Sparks `node-a` and `node-b`, tailnet `100.64.0.x`,
LAN `192.0.2.x`. The example config's node ids are `a` and `b`. Those ids are
what `SPARKSI_NODE_ID` must be; they are not hostnames. A one-Spark site has
only `node-a`; a third or fourth Spark repeats the `node-b` steps with its own
id.

## Controller user unit

On the controller, as the user that will run the UI:

```bash
git clone <repo-url> "$HOME/src/sparksi" && cd "$HOME/src/sparksi"
python3 -m venv .venv && .venv/bin/pip install -e .
mkdir -p ~/.config/sparksi ~/.config/systemd/user
cp config/sparksi.example.toml ~/.config/sparksi/sparksi.toml   # edit
cp config/catalog.example.yaml ~/.config/sparksi/catalog.yaml   # edit
.venv/bin/sparksi install-unit --print --unit controller \
  --prefix "$HOME/src/sparksi" \
  > ~/.config/systemd/user/sparksi-controller.service
```

Review the file before loading it. `ExecStart` is `sparksi serve` with a port
and no bind address. The server binds `[controller] bind` from
`sparksi.toml`, or the catalog's `controller.bind_allow` when `catalog.yaml`
has a `controller:` block. The built-in default is loopback `127.0.0.1` plus
the tailnet address `100.64.0.1`. `0.0.0.0`, `::` and `*` are refused; which
deny lists apply is in [config.md](config.md#listen-addresses).

Loading and enabling the unit is a separate step you run yourself:

```bash
systemctl --user daemon-reload
systemctl --user enable --now sparksi-controller.service
```

Dry check first on a spare port, loopback only:
`.venv/bin/sparksi serve --port 7510`, then diff `/api/status`, `/api/monitor`
and `/api/fleet` against the running instance.

An existing `dgx-mem/` state directory is reused automatically. To point an
old unit at this tree, use a drop-in that overrides `WorkingDirectory`,
`Environment`, and `ExecStart`. Rolling back is deleting the drop-in,
`daemon-reload`, and restarting.

## Controller service name

`packaging/systemd/sparksi.service` is the user unit for the controller.
Copy it into `~/.config/systemd/user/` and enable it yourself when you want
that name. No install script copies the file, enables the unit, or disables
a controller unit that is already running.

A unit that is already enabled keeps running. The cut-over is a hand step:
disable that unit, install `sparksi.service`, then enable the new one. Do
not enable both. They would start two controllers.

## Node announcer user unit

The announcer script holds no node id and no address. Those come from the
environment when the unit starts:

- `SPARKSI_NODE_ID` — a `[[nodes]] id` from `sparksi.toml` (`a` or `b` in the example)
- `SPARKSI_CONTROLLER` — `ip:port` of the controller announce listener
  (example `100.64.0.1:7501`)
- `SPARKSI_ADVERTISE` — that node's `tailscale` address from the same
  `[[nodes]]` entry. `sparksi install-unit --print` writes it into the
  unit. The announcer GETs `http://that-address:port/v1/models` once, with
  a 0.4 second limit and no redirect or proxy, and publishes a wildcard
  URL only when the served id matches. An empty value publishes nothing.
  Printing the unit does not enable or start the announcer.
- `SPARKSI_PROTECT` — the protection presets active on the controller when
  the unit is rendered (empty when none). The announcer cannot hear from the
  controller, so this line is how a node learns to skip the video service.
  When updating an existing install whose unit predates it, add
  `Environment=SPARKSI_PROTECT=video-service` to each node's announcer unit
  (or re-render the unit) before restarting the announcer. See
  [`[protect]`](config.md#protect).

Copy the script to node-a. This keeps `sparksi-announce.bak` if a copy is
already there, and it does not restart the service:

```bash
scripts/install-announce.sh --host node-a
```

Pass `--restart` only when you want `systemctl --user restart
sparksi-announce.service` (or `dgx-announce.service` where a node has it) afterwards. The script never enables the unit.

Render the unit (again, print only) and install it on the node yourself:

```bash
sparksi install-unit --print --unit announce \
  --node-id a --controller 100.64.0.1:7501
```

The unit runs `/usr/bin/python3 %h/.local/bin/sparksi-announce`. Repeat for
node-b with `--node-id b` and the same controller address, and for any third
or fourth Spark with its own id.

## Firewall

The controller listens for announce datagrams on UDP 7501 (the
`announce_port` in `sparksi.toml`). Allow that port only from the nodes'
tailnet addresses, not from the LAN:

```
# example packet filter on the controller
udp dport 7501 ip saddr { 100.64.0.11, 100.64.0.12 } accept
udp dport 7501 drop
```

`100.64.0.11` is node-a and `100.64.0.12` is node-b in the example config.
Add each further Spark's tailnet address to the same set.
Do not open `192.0.2.0/24` to this port. The announcer does not listen; only
the controller does.

## SSH

The controller reaches each node with a key, in BatchMode, so a missing key
fails instead of prompting:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 node-a true
ssh -o BatchMode=yes -o ConnectTimeout=8 node-b true
```

Probe, stage, drop, and recipe runs use the same options. The `ssh` value in
each `[[nodes]]` entry is the destination (`node-a`, `node-b`, or `local` when
the controller runs on that Spark).

## Passwordless sudo

Apply and stop run `sudo -n /bin/systemctl` with action `start`, `stop`, or
`restart` (and sometimes `--no-block`) and a unit name taken from the
catalog. Any other sudo command is refused. Grant the node user passwordless
sudo for those three actions only, not for enable, disable, or a shell:

```
<node-user> ALL=(root) NOPASSWD: /bin/systemctl start *, /bin/systemctl stop *, /bin/systemctl restart *
```

`sudo -n` means a missing rule fails immediately. It does not ask for a
password.

## Optional metrics

`[controller] metrics_url` (example `http://127.0.0.1:8428`) is an optional
VictoriaMetrics address for the day token ledger. Leave it unused if you do
not run that service. `contrib/token-discover` can refresh scrape targets;
it is not installed or started by the units above.
