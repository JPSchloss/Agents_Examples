# Data Engineering Principles (house style)

Guidance the team follows. Agents should cite and apply these when advising or building.

## Pipeline design

- **Idempotency:** running a pipeline twice on the same input must produce the same
  output and must not duplicate data. Prefer full-refresh `replace` for small tables, or
  upserts keyed on a natural id (`order_id`) for incremental loads. Never blind-append.
- **Medallion layers:** organize transformations as Bronze (raw, as-ingested) → Silver
  (cleaned, typed, deduped) → Gold (business-level aggregates serving dashboards). Keep
  each layer reproducible from the one before it.
- **Schema on write for serving:** dashboards read from the Gold layer with explicit
  types, so the front end never has to clean data at render time.
- **Small, verifiable steps:** profile → clean → validate → load. Validate with explicit
  checks (row counts, null rates, value domains) rather than eyeballing.

## Data quality checks to run after cleaning

1. Primary key uniqueness (`order_id`).
2. No nulls in required columns (`order_id`, `order_date`, `quantity`, `unit_price`).
3. Value-domain checks (`region` ∈ canonical set, `category` ∈ canonical set).
4. Range checks (`quantity` > 0 for revenue rows).
5. Reconciliation (`total` ≈ `quantity` × `unit_price`).

## Reliability

- Make transforms **deterministic**; avoid depending on row order.
- Fail loudly: a cleaning script should raise/print when an assumption is violated, not
  silently coerce bad data.
- Keep raw inputs immutable; write only to a separate workspace/output location.

## Front-end / serving

- Dashboards should read from a prepared store (SQLite/warehouse table), not re-parse CSVs.
- Show data freshness and row counts so users can sanity-check what they're looking at.
- Defensive rendering: handle empty result sets without crashing.
