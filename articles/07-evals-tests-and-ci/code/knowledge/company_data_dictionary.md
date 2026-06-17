# Acme Analytics — Sales Data Dictionary & Business Rules

This is the authoritative definition of the `sales` dataset. Agents should treat these
rules as ground truth when cleaning or interpreting the data.

## Columns

| Column | Meaning | Type | Rules |
|--------|---------|------|-------|
| `order_id` | Unique id for an order line | integer | Must be unique. Duplicate `order_id` rows are accidental double-exports and must be dropped. |
| `order_date` | Date the order was placed | date (ISO `YYYY-MM-DD`) | Source data mixes formats; always normalize to ISO. |
| `customer` | Customer company name | string | Trim whitespace. Preserve original casing for display. |
| `region` | Sales region | category | Canonical values are exactly: **North, South, East, West**. Normalize casing. A blank region means "Unassigned". |
| `product` | Product name | string | Known products: Widget Pro, Widget Lite, Gizmo, Service Plan. |
| `category` | Product category | category | Canonical values: **Hardware, Services**. Normalize casing. Missing category can be inferred from `product` (Service Plan → Services, everything else → Hardware). |
| `quantity` | Units ordered | integer | Must be a positive integer. |
| `unit_price` | Price per unit in USD | float | Source data sometimes includes a `$` symbol; strip it and parse as float. |
| `total` | Line revenue in USD | float | **Business rule: `total` = `quantity` × `unit_price`.** When `total` is missing, recompute it. When it disagrees with the formula, the formula wins. |

## Business rules

- **Returns:** a negative `quantity` represents a return/correction. For *revenue*
  reporting these rows should be **excluded** (or handled in a separate returns measure),
  never silently kept as negative revenue.
- **Revenue** = sum of `total` over valid (non-return) rows.
- **Region "Unassigned"** rows are valid and should appear in dashboards as their own
  bucket, not dropped.
- The fiscal reporting grain is one row per `order_id`.

## Definition of "clean"

A cleaned `sales` table satisfies: unique `order_id`; ISO dates; `region` and `category`
in their canonical value sets; numeric `quantity`/`unit_price`/`total`; no return rows in
the revenue table; `total` consistent with the formula.
