@workspace /new You are an elite AI Architect and Senior Python Developer. Generate a complete, production-ready, modular Python application for the "Lumen Zero-Touch NaaS Broker".

Generate the code in a single shot without pausing. Output the complete file structure, the required Python modules, and a comprehensive README.md. The code must be bug-free, strictly typed, and PEP-8 compliant.

### 1. Project Context & Business Logic

- **Purpose:** Eliminate the human sales engineering bottleneck in telecom provisioning. Convert natural language enterprise network requests directly into billable, provisioned API payloads.
- **Environment Constraint (MOCK ONLY):** This is a mock project. DO NOT make any real HTTP requests to live Lumen APIs. You must create a mock API function within the application that accepts the final JSON payload, validates it, and simulates a successful 200 OK provisioning response.
- **Budget Constraint:** The developer has a strict $20 API limit. Use cost-efficient models dynamically.

### 2. Tech Stack & Model Selection

- **Frontend UI:** `streamlit` (for conversational UI and Mermaid.js network topology rendering).
- **Backend & GenAI:** Python 3.11+, `openai` SDK, `pydantic` (for strict schema validation).
- **Model Routing:**
  - **Conversational NLU & BOM Generation:** Use `/v1/chat/completions` with `gpt-5-mini` (highly cost-effective for parsing intents like location, bandwidth, and destination).
  - **API Payload Generation:** Use `/v1/responses` with `gpt-5.3-codex` (strictly for generating the final provisioning JSON upon user approval).

### 3. Innovative Features to Include

- **Automated Multi-Cloud Cost Optimization:** If a user asks to connect to a cloud but doesn't specify which one, the agent dynamically recommends the cheapest egress path (AWS vs. GCP vs. Azure) based on local context data.
- **Visual Topology Rendering:** The UI must generate and display a Mermaid.js diagram of the proposed network architecture (Source Node -> Lumen Fabric -> Cloud Provider) alongside the quote.
- **Self-Healing JSON Validation & Mock Execution:** The system must use Pydantic to validate the Codex-generated payload against the mock Lumen API schema. Once validated, it passes the payload to a local mock API function that returns a simulated transaction ID and success state.

### 4. Required File Structure & Module Responsibilities

Generate the following separate files:

- **`config.py`**: Environment variables, API keys, and model constants (`gpt-5-mini`, `gpt-5.3-codex`).
- **`lumen_context.json`**: A mock dataset containing Lumen on-net locations (e.g., Seattle, Chicago, Hyderabad), hourly pricing for 1Gbps/10Gbps ports, and DDoS add-on costs. Include a strict JSON schema representation for the Lumen `POST /Customer/v3/Ordering/orderRequest` endpoint.
- **`mock_api.py`**: A module containing `mock_lumen_provisioning_api(payload: dict) -> dict`. This function accepts the generated JSON, simulates a 2-second network delay using `time.sleep`, logs the payload, and returns a mock JSON response with `{"status": "SUCCESS", "transaction_id": "LUMEN-NAAS-12345", "message": "Network provisioned"}`.
- **`tools.py`**: Functions to load the local context, query available ports (simulating a $0 CapEx check), and calculate the total hourly quote (Bill of Materials).
- **`agent.py`**: The core LLM orchestration logic.
  - `generate_quote()`: Calls `gpt-5-mini` to extract intent and format the BOM.
  - `generate_mermaid_diagram()`: Generates the visual topology string.
  - `generate_provisioning_payload()`: Calls `gpt-5.3-codex` when the chat history contains "QUOTE_APPROVED" to generate the strict Pydantic-validated JSON, which is then passed to `mock_api.py`.
- **`app.py`**: The Streamlit entry point. Manages session state, renders chat bubbles, displays the BOM table, renders the Mermaid diagram, and provides an "Approve Quote" workflow.
- **`requirements.txt`**: `streamlit`, `openai`, `pydantic`, `python-dotenv`.

### 5. README.md Requirements

Generate a highly professional `README.md` that contains:

- **Project Title:** Lumen Zero-Touch NaaS Broker
- **Executive Summary:** What the project is and the core objective.
- **The Problem Solved:** Detailed explanation of the traditional manual quoting/provisioning drawbacks (sales delays, human API translation errors, high friction).
- **Revenue Benefits to Lumen:** How this accelerates time-to-revenue and drives NaaS consumption.
- **Architecture & Flow:** Explanation of how `gpt-5-mini` handles NLU, `gpt-5.3-codex` handles the payload execution, and how the Mock API simulates the physical network layer safely.
- **Setup Instructions:** How to install dependencies, set API keys, and run the Streamlit app locally.

### 6. Strict Code Constraints

- Do not include placeholder text like "Add your logic here"; write the actual functional logic for a local prototype.
- Keep the code extremely crisp. Do not over-engineer; use functional programming patterns where appropriate.
- Ensure full error handling for API call failures or Pydantic validation errors, displaying user-friendly error messages in the Streamlit UI.
