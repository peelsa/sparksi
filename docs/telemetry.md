# Telemetry files

The overview chart shows a 20-minute window that can move through the last
24 hours. The files under the state directory keep every fresh sample the
controller has collected since this archive was turned on. They are not
trimmed when the chart moves, and nothing in Sparksi deletes them.

Samples from before the archive existed were not recorded. Starting the
archive does not reconstruct them.

## Where the files are

Next to `history.json` in the controller state directory:

```text
telemetry/schema.json
telemetry/days
telemetry/error
telemetry/<role>.recent.json
telemetry/<role>/YYYY-MM-DD.jsonl
telemetry/<role>/YYYY-MM-DD.tip.json
telemetry/<role>/YYYY-MM-DD.idx
```

`<role>` is `head`, `worker`, `node3` or `node4`, one directory per configured
Spark. Roles are not host names. A line does not include a
host name, address, account, or path. The day is the sample's UTC date.

`history.json` is only the hot cache. It keeps about one hour and drops
older rows. The JSONL files are the durable copy. `days` gains one line
when a day file is created. A history query never lists that catalog: a
window of at most 24 hours is one or two dated paths. `.tip.json` is one
small object for that day. It is replaced only when its recorded size still
matches the sample file. If it does not, the count is unknown and coverage
comes from the first and last complete sample lines. `.idx` is an offset
about once a minute. `<role>.recent.json` is the
role summary the dashboard shows, independent of which catalog page was
requested. Nothing walks every stored day to answer a query or a page.

## Record

Each line is one JSON object. `schema.json` is the same document
`GET /api/telemetry` returns as `schema`.

```json
{"cpu_pct":4,"gpu_pct":12,"gpu_temp_c":42.2,"mem_pct":25,"mem_used_gib":32,"power_w":40,"role":"head","t":1700000000,"tok_s":18}
```

| Field | Meaning |
|---|---|
| `t` | integer UTC epoch seconds |
| `role` | `head`, `worker`, `node3` or `node4` |
| `gpu_pct`, `cpu_pct`, `mem_pct` | busy or in-use, percent |
| `mem_used_gib` | unified memory in use, gibibytes |
| `gpu_temp_c` | GPU temperature, celsius. The chart converts to Fahrenheit |
| `tok_s` | decode tokens per second for models on that role |
| `power_w` | board power, watts |

A missing key was not reported. A present `0` is a real zero. Gaps stay
gaps. Collection writes a line only for a fresh report, from the same pass
that updates the hot cache, not from a chart GET.

## Reading them

`GET /api/telemetry` is read-only and paged. `limit` defaults to 30 and
cannot exceed 50. `role` is one of the configured roles. `cursor` is the previous
page's `next` value, a byte offset into `telemetry/days`. Each request reads
only that slice, not every day name. `roles` is the summary for each configured Spark
and does not depend on which page you asked for. A bad page is HTTP 400.

`GET /api/history?machine=head&range=1h` still returns `{machine, range, now, samples}`.
`machine` may be a role (`head`, `worker`, `node3`, `node4`) or a configured node id. Optional
`start` and `end` are UTC epoch seconds and select that window instead of
the named range. The span cannot exceed 24 hours. `NaN`, infinities, and
timestamps outside 1970 through 2100 are HTTP 400. Optional `since` is a
UTC epoch second. Samples at or before it are left out. The named range, or
`start` and `end` when both are set, still bounds the read, so `since` does
not widen a query. A query that omits `since` is the same window as before.
A bad `since` is HTTP 400. A window of an hour or less is returned raw. A
wider window, still at most 24 hours, is averaged into at most about 1200
buckets. Empty stretches are omitted, so a wide query can hide a gap shorter
than one bucket. Each timestamp counts once. If the hot cache and the file
both have that second, the hot sample is the one that is averaged.

The overview chart seeds the last 20 minutes once per machine. A contiguous
live ring does not call `/api/history`. `GET /api/monitor` adds `history`,
the newest hot sample for each machine id, and the page appends that sample
to a 20-minute ring kept in memory. Each machine keeps a cursor on the last
timestamp that has no hole after it. A newer sample does not move the cursor
past a hole. The page asks for `since` that cursor until a history response
covers the machine, including when an earlier read failed or another read
was already in flight. A fresh sample into an empty ring, or a jump wider than 20 minutes, seeds
that machine again. A monitor sample older than the live window is ignored,
and so is the same timestamp repeated while the machine is quiet. The live
edge is the later of the client clock and the newest sample, so a client
clock behind the server does not drop the point. The chart pauses while the
document is hidden. Coming back, or returning from a pan to live, asks for
`since` only when the cursor is still behind the newest sample. One history read is in flight at a time. Panning reads
that older 20-minute window and keeps a few of those windows in memory, so
moving back to one does not download it again. Returning to live uses the
ring. The page does not write the samples to browser storage. The one-hour
hot ring on the server stays in place for other callers.

A live hour that the process has already matched to the files is answered
from memory. The next polls do not reread the day file. An older slice
seeks to the last minute-offset at or before the window and stops at the
end of the window. Several viewers share that behavior. They do not each
scan the retained history.

Both calls only read. A torn tail is left in the file. The next append
puts a newline after it so the new sample is its own line. The torn bytes
stay. The same separation is used for a torn offset line and a torn catalog
line. A tip or offset write that fails does not remove the sample. The next
successful append does not pretend the missed count is 1. The count stays
unknown, and discovery takes the first and last timestamps from the sample
file. The sample itself is still returned by a history query. `last_error`
is removed after a later append finishes.

## Writes, restarts, and disk errors

The day is recorded in `telemetry/days` before any sample from that day is
written. A crash after the record and before the sample leaves an empty day.
Discovery still lists it, with empty timestamps, and does not scan the
directory. The completion marker is written only after that catalog line is
durable. The checkpoint moves only through lines that already have markers,
so one role cannot skip another role's unmarked line. A later append finishes
the missing marker and does not write the day twice. The tip is replaced
after the day is recorded, and only when the previous tip still describes
every earlier byte. About once a minute the collector appends an offset. A torn tip or offset line is left in place and separated
before the next record, the same way a torn sample line is. A restart loads
the hot hour from `history.json` and leaves the JSONL files in place. A
range is read from the archive only when that exact interval has not been
loaded yet. An empty stretch is still a loaded interval. A later disjoint
range is read again. Once an interval has been read, repeating it does not
reread the files.

If the disk is full or the directory cannot be written, the controller keeps
running. The sample remains in the hot hour when that write succeeded, and
`last_error` on `GET /api/telemetry` says the durable write failed. The
error text does not include a filesystem path.

## Growth

One fresh report per second from two Sparks is on the order of 25 MB per day,
or roughly 9 GB per year, plus the small index. It scales with the number of
Sparks. That is unbounded on
purpose. Move or copy old day files yourself if you need the space. If a
file is removed, those samples are gone. The software will not recreate
them and will not delete the ones that remain.

The chart does not download a full day into the browser. A live second does
not download the hour either. A 20-minute pan reads that slice. A 24-hour
request is downsampled.
