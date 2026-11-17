# Security

- The controller listens on `[controller] bind` (by default `127.0.0.1` plus
  one tailnet address), or on the catalog `controller:` block's `bind_allow`.
  `0.0.0.0`, `::` and `*` are always refused. Any other address, LAN included,
  is refused only when it is in a `bind_deny` list, so list each machine's LAN
  and WiFi addresses there. See
  [Listen addresses](docs/config.md#listen-addresses).
- `sparksi serve --port` refuses 8000, 8001, 8004, 8007, 8011 and any
  catalog `host.reserved_ports`. Recipe ports must be in 8010-8099 and are not
  in a `reserved_ports` list. With the `video-service` preset on (off by
  default; see [`[protect]`](docs/config.md#protect)), port 8188 is refused
  everywhere too.
- Mutating API calls need the lock OFF or the unlock token. The token lives in
  the runtime state dir (`$XDG_RUNTIME_DIR/sparksi/unlock.token`), is never
  returned by GET routes, and must never be committed.
- Node operations run over SSH with `BatchMode=yes`. The node user needs
  passwordless sudo only for `systemctl start|stop|restart` of the units named
  in the catalog.
- Announce packets are accepted only from the tailnet addresses configured per
  node. Firewall the announce port to those addresses as well.

Report security issues privately to the Peelsa maintainers.
