#!/usr/bin/env bash
set -euo pipefail
PROMPT="${1:-"Rochester Auto Company (RAC) produces four types of cars: subcompact, compact, intermediate, and luxury. RAC also produces trucks and vans.

Vendor capacities limit total production capacity to, at most, 1.2 million vehicles per year.

Subcompacts and compacts are built together in a facility with a total annual capacity of 620,000 cars.

Intermediate and luxury cars are produced in another facility with capacity of 400,000.

The truck/van facility has a capacity of 275,000.

RAC's marketing strategy requires that subcompacts and compacts must constitute at least half of the product mix for the four car types.

The Corporate Average Fuel Economy (CAFE) standards in the Energy Policy and Conservation Act require an average fleet fuel economy of at least 27 mpg.

Profit margins, market potential, and fuel efficiencies are summarized below:

**Subcompact:** Profit Margin: $150/vehicle, Market Potential: 600,000 sales, Fuel Economy: 40 mpg

**Compact:** Profit Margin: $225/vehicle, Market Potential: 400,000 sales, Fuel Economy: 34 mpg

**Intermediate:** Profit Margin: $250/vehicle, Market Potential: 300,000 sales, Fuel Economy: 15 mpg

**Luxury:** Profit Margin: $500/vehicle, Market Potential: 225,000 sales, Fuel Economy: 12 mpg

**Truck:** Profit Margin: $400/vehicle, Market Potential: 325,000 sales, Fuel Economy: 20 mpg

**Van:** Profit Margin: $200/vehicle, Market Potential: 100,000 sales, Fuel Economy: 25 mpg

What is the optimal profit for RAC?"}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/orlm-qwen-mini}"

cd "$(dirname "$0")/../ORLM"
python3 scripts/inference.py \
  --model_name_or_path "$SAVE_PATH" \
  --tensor_parallel_size 1 \
  --prompt "$PROMPT"
