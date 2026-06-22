# How AI Can Predict Road Collision Severity — And Why It Matters for UK Road Safety

## I analysed 100,000+ real collision records to uncover what actually makes roads dangerous — and built a live tool anyone can use to explore the data

---

In 2024, over **100,000 road collisions** were reported across the UK, resulting in more than **128,000 casualties** and **183,000+ vehicles** involved. Behind every statistic is a family affected, an emergency service stretched, and a community impacted.

But here's the thing — most of these collisions follow patterns. Certain roads, certain times, certain conditions produce far worse outcomes than others. What if we could see those patterns clearly, and use them to make smarter decisions about where to invest in road safety?

That's exactly what I set out to build — and you can explore it right now.

---

## Try It Live

I've deployed the platform across three hosting services so anyone can access it:

| Platform | Link |
|----------|------|
| **Streamlit Cloud** | [uk-road-collision-intelligence.streamlit.app](https://uk-road-collision-intelligence-imutd2qu9ybzujv9appkmd.streamlit.app/) |
| **Hugging Face Spaces** | [huggingface.co/spaces/JUMMAMOHAMMAD477](https://huggingface.co/spaces/JUMMAMOHAMMAD477/uk-road-collision-intelligence) |
| **Render** | [uk-road-collision-intelligence.onrender.com](https://uk-road-collision-intelligence.onrender.com) |

No installation needed. Just click any link and start exploring.

---

## What I Built

The **UK Road Collision Intelligence Platform** takes raw government data and transforms it into actionable insights through three layers:

1. **An interactive dashboard** where anyone can explore collision patterns by location, time, weather, road type, and more — with real-time filters
2. **A geographic hotspot map** that pinpoints the **2,335 most dangerous locations** across the UK, with 117 flagged as critical
3. **A prediction tool** that estimates how severe a collision would be given specific road and environmental conditions — powered by AI that was tested against four different approaches to find the most reliable one

No coding required to use it. Just open the dashboard, apply filters, and explore.

---

## The Numbers That Matter

Before diving into how it works, here are the headline findings that came straight from the data:

| Finding | What the Data Shows |
|---------|-------------------|
| Night driving | Fatal collision rate is **2.1x higher** than daytime |
| Rural vs urban roads | Fatal rate is **3.4x higher** in rural areas |
| Speed limit impact | 60mph zones have a **6.6x higher** fatal rate than 20mph zones |
| Heavy goods vehicles | HGV-involved collisions have a **5.96% fatality rate** — 4x the national average |
| Elderly casualties (65+) | **3.58% fatal rate** — 2.4x the average across all ages |
| Dangerous hotspot clusters | **2,335 identified**, with **117 rated Critical** |
| Most dangerous hour | Collisions peak at **5 PM** (8,784 in 2024), but fatal collisions peak between **8 PM and midnight** |

These aren't assumptions — they're facts drawn from every reported collision in the UK for 2024.

---

## What the Data Reveals

### The most dangerous time isn't when you'd expect

Collisions are most *frequent* during the evening rush hour (4–6pm) — that's when traffic is heaviest. But **fatal** collisions peak much later, between **8pm and midnight**. Weekend nights are especially deadly.

This matters for resource planning. If you're allocating emergency services or police patrols purely based on collision volume, you're missing the window when the most serious incidents happen.

### Speed is the single biggest factor

The fatality rate rises dramatically once the speed limit exceeds 60mph. Rural single carriageways — country roads with the national speed limit — are the most dangerous road type by a wide margin.

This isn't surprising, but the data quantifies it precisely. It gives councils and transport authorities the evidence base to justify speed limit reviews on specific roads.

### Darkness and rural roads are a deadly combination

Dark conditions without street lighting produce the highest fatal collision rate of any lighting condition. When you combine darkness with a rural location, the risk compounds — this combination is the **single strongest predictor** of a fatal outcome in the entire dataset.

This has direct implications for infrastructure investment. Street lighting on high-risk rural stretches could save lives — and the data can identify exactly which stretches to prioritise.

### Weather matters less than people think

Fine, dry weather actually sees the **most collisions** — simply because more people drive in good conditions. The fatal *rate* does increase in fog and heavy rain, but the absolute numbers are smaller. The real danger isn't bad weather — it's darkness, speed, and isolation.

### Vulnerable road users face the worst outcomes

Motorcycle involvement significantly increases collision severity. Collisions involving heavy goods vehicles result in more casualties per incident — with a fatality rate 4x the national average. Elderly casualties face disproportionately severe outcomes. E-scooter-related collisions are a growing and concerning new category.

---

## How the Hotspot Map Works

Not every stretch of road is equally dangerous. Using geographic clustering, the platform identifies **2,335 collision hotspots** — locations where incidents cluster within a 500-metre radius. **61.5% of all collisions fall within these hotspots.**

Each hotspot is scored and ranked using four factors:

- **Fatality rate** — what proportion of collisions at this location are fatal
- **Serious injury rate** — what proportion result in serious injuries
- **Environmental danger** — how often collisions happen in dark, wet, or high-speed conditions
- **Volume** — how many collisions occur at this location

Hotspots are classified into four risk tiers:

| Tier | Count | What It Means |
|------|-------|--------------|
| **Critical** | 117 | Highest risk — 9.9% fatality rate (6.6x national average) |
| **High** | 336 | Significantly above average risk |
| **Medium** | 706 | Moderate risk requiring monitoring |
| **Low** | 1,176 | Below average risk but still clustered |

The top 100 most dangerous hotspots are plotted on an interactive map where stakeholders can zoom in, click on clusters, and see the complete risk profile for any location.

### Who benefits from this?

- **Local councils** can identify which roads in their jurisdiction need safety interventions
- **Police forces** can see which hotspots fall within their area and how they compare nationally
- **Transport planners** can prioritise infrastructure spending based on data, not assumptions
- **Insurance companies** can better understand geographic risk concentrations
- **Road safety campaigners** can use specific data to support their advocacy

---

## The Prediction Tool: "What If" Scenario Planning

The platform includes a severity prediction tool that answers a practical question: **given a specific set of conditions, how severe would a collision likely be?**

Users select from dropdown menus and sliders:

- **Road characteristics** — speed limit, road type, road class, urban or rural
- **Environmental conditions** — lighting, weather, road surface, time of day, day of week
- **Collision details** — number of vehicles, casualties, vehicle types involved
- **People involved** — driver age, casualty age, vehicle age

Hit "Predict" and the system returns:

- A severity prediction: **Fatal**, **Serious**, or **Slight**
- The probability for each outcome (e.g., 3% Fatal, 22% Serious, 75% Slight)
- A list of identified risk factors in plain language (e.g., "High speed zone," "Dark conditions," "Rural area")

### Why this matters for decision-makers

This isn't about predicting individual crashes — it's about **scenario modelling**. A transport authority could ask:

- *"If we reduce the speed limit on this rural road from 60mph to 40mph, how does the predicted severity change?"*
- *"If we install street lighting on this unlit stretch, what's the impact on fatal collision probability?"*
- *"Which combination of conditions produces the highest risk, and where do those conditions exist on our road network?"*

The prediction tool turns data into a planning instrument.

---

## Four AI Models Tested — One Clear Winner

I tested four different machine learning approaches to find the most reliable predictor. Here's what mattered:

| Approach | What It's Good At | Fatal Detection Rate |
|----------|-------------------|---------------------|
| Logistic Regression | Simple baseline | 57% (but poor overall accuracy) |
| Random Forest | Good overall accuracy | Only 8% |
| XGBoost | Strong all-round performer | 11% |
| **LightGBM (Tuned)** | **Best balance of accuracy and fatal detection** | **35%** |

**Why fatal detection matters most:** Only ~2% of collisions are fatal. A naive model could achieve 98% "accuracy" by simply never predicting a fatal outcome. That looks great on paper but is completely useless for road safety. LightGBM was specifically chosen because it catches 4x more fatal collisions than Random Forest while maintaining strong overall performance.

The winning model was then fine-tuned through **50 rounds of automated optimisation** to squeeze out maximum performance — achieving the highest overall reliability score of all models tested.

### What drives the predictions?

The top factors the model relies on:

1. **Number of casualties** — more people involved means more severe outcomes
2. **Speed limit** — higher speed limits strongly correlate with fatality
3. **Casualty age** — elderly people face significantly worse outcomes
4. **Vehicle size differences** — large engine capacity differences suggest car-vs-truck scenarios
5. **Composite danger score** — a combined measure of darkness, weather, surface, speed, and rural location
6. **Dark rural roads** — the interaction of these two factors is more dangerous than either alone

These aren't black-box outputs — they align with what road safety experts already know, which gives confidence that the model is learning real patterns, not statistical noise.

---

## The Dashboard: Insights at Your Fingertips

The platform is accessed through an interactive web dashboard with five sections:

### Overview
At-a-glance KPIs — total collisions, fatal count, serious injuries, total casualties. Plus charts showing how collisions distribute by hour, day of week, month, and speed limit. Everything updates instantly when you apply filters.

### Hotspot Map
Two interactive maps: one showing individual collisions colour-coded by severity (Fatal in red, Serious in orange, Slight in green), and one showing hotspot clusters sized by collision count and coloured by risk tier. Zoom into any area to see local patterns.

### Conditions Analysis
Side-by-side comparisons of how fatal rates change across light conditions, weather, road surface, and overall danger score. Immediately reveals which environmental factors matter most.

### Model Performance
Transparent comparison of all four models tested, plus a visual breakdown of which features matter most for predictions. Full accountability — no hidden decisions.

### Predict Severity
The interactive scenario planning tool described above. Select conditions, click predict, and get an instant severity estimate with probabilities.

### Sidebar Filters
Every analytical page can be filtered by severity, urban/rural, month range, hour range, road class, speed limit, and weather. A counter shows how many records match your current selection out of the full 100,927.

---

## How It's Built (The Short Version)

For those curious about the technical foundation — without getting into the code:

```
Raw Government Data (3 tables, 412,000+ records)
    ↓
Data Cleaning & Preparation → 112 engineered features
    ↓
Geographic Hotspot Detection → 2,335 danger clusters identified
    ↓
AI Model Training → 4 models tested, best one selected and fine-tuned
    ↓
Live Dashboard + Prediction API → deployed across 3 platforms
    ↓
Automated Monitoring → detects when data patterns change
```

The entire pipeline is **automated and reproducible** — when the DfT releases new data next year, the system can be retrained in minutes.

The platform also includes:
- **74 automated tests** to ensure data quality and model reliability
- **Continuous integration** that re-runs the full pipeline on every update
- **Drift detection** that alerts when incoming data looks different from what the model was trained on
- A **REST API** for organisations that want to integrate predictions into their own systems

---

## The Business Case

### For Local Authorities and Transport Bodies
- Evidence-based prioritisation of road safety interventions
- Identify which specific roads and junctions in your area are the most dangerous
- Model the impact of proposed changes (speed limits, lighting, road redesign) before spending
- Quantified data to support funding applications

### For Emergency Services
- Allocate resources based on when and where the most *severe* incidents occur, not just the most *frequent*
- Identify temporal patterns for shift planning
- Understand which conditions most often lead to fatal responses

### For Insurance Companies
- Understand geographic and environmental risk concentrations
- Improve risk assessment models with data-driven severity predictions
- Identify emerging trends (e.g., e-scooter-related claims)

### For Road Safety Researchers and Campaigners
- Access to clear, visual evidence of which factors drive fatal outcomes
- Data to support policy recommendations with quantified impact
- Transparent methodology — the full source code is open for review

### For the Public
- Transparent access to road safety data for their local area
- Understanding of which conditions make driving most dangerous
- Empowerment through information

---

## What's Next

This platform is a foundation. Future enhancements could include:

- **Plain-language explanations** for every prediction — not just "Serious," but specifically *why* the model thinks so
- **Weekly forecasting** — predicting how many collisions to expect in each region next week
- **Live data feeds** — connecting to real-time police incident reports instead of annual data releases
- **Before-and-after analysis** — measuring whether a specific road intervention actually reduced severity
- **Regional comparison reports** — automated briefings for each police force or local authority area

---

## The Bottom Line

100,000 collisions a year aren't random. They follow patterns shaped by speed, darkness, road design, time of day, and location. This platform makes those patterns visible, quantifiable, and actionable.

The data is already being collected. The question is whether we use it.

---

## Explore the Platform

| Resource | Link |
|----------|------|
| **Live Dashboard (Streamlit)** | [uk-road-collision-intelligence.streamlit.app](https://uk-road-collision-intelligence-imutd2qu9ybzujv9appkmd.streamlit.app/) |
| **Live Dashboard (Hugging Face)** | [huggingface.co/spaces/JUMMAMOHAMMAD477](https://huggingface.co/spaces/JUMMAMOHAMMAD477/uk-road-collision-intelligence) |
| **Live Dashboard (Render)** | [uk-road-collision-intelligence.onrender.com](https://uk-road-collision-intelligence.onrender.com) |
| **Source Code (GitHub)** | [github.com/jumma786/uk-road-collision-intelligence](https://github.com/jumma786/uk-road-collision-intelligence) |
| **Data Source** | [DfT Road Casualty Statistics 2024](https://www.data.gov.uk/dataset/road-accidents-safety-data) |

---

## About the Author

**Jumma Mohammad Teli** — Data Scientist and ML Engineer passionate about using data to solve real-world problems. This project combines machine learning, geospatial analysis, and MLOps engineering to turn publicly available government data into a practical tool for road safety.

Connect with me:
- **Email:** jummamohammad477@gmail.com
- **GitHub:** [github.com/jumma786](https://github.com/jumma786)
- **Hugging Face:** [huggingface.co/JUMMAMOHAMMAD477](https://huggingface.co/JUMMAMOHAMMAD477)

If you're a council, transport body, researcher, or organisation working on road safety — I'd love to hear how this data could help your work.

---

*Data source: UK Department for Transport — Road Casualty Statistics 2024. Built with Python, LightGBM, Streamlit, FastAPI, and Plotly. The platform is open source under the MIT licence.*
