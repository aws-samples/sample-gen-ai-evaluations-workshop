"""Synthetic MCP server fixtures for tool-calling evaluation.

Provides realistic MCP tool schemas, error responses, and permission sets
for testing AI agent behavior. No live MCP server required — the notebook
uses these fixtures to evaluate tool-call quality offline.

Designed to be imported by the evaluation notebook.
"""

from __future__ import annotations


# =============================================================================
# TOOL SCHEMAS (tools/list response)
# =============================================================================

MCP_TOOLS = [
    {
        "name": "get_inventory_levels",
        "description": "Get current on-hand inventory for a SKU at a specific distribution center",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU identifier (e.g., CHIPS-BBQ-12OZ)"},
                "location": {"type": "string", "description": "Distribution center code (e.g., DC-DALLAS)"},
            },
            "required": ["sku"],
        },
    },
    {
        "name": "get_inventory_history",
        "description": "Get historical inventory movements for a SKU over a date range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU identifier"},
                "start_date": {"type": "string", "format": "date", "description": "Start of date range (YYYY-MM-DD)"},
                "end_date": {"type": "string", "format": "date", "description": "End of date range (YYYY-MM-DD)"},
            },
            "required": ["sku", "start_date", "end_date"],
        },
    },
    {
        "name": "get_demand_forecast",
        "description": "Get forward demand forecast for a SKU at a location over a specified horizon",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "location": {"type": "string"},
                "horizon_weeks": {"type": "integer", "minimum": 1, "maximum": 52},
            },
            "required": ["sku", "location", "horizon_weeks"],
        },
    },
    {
        "name": "get_supplier_status",
        "description": "Get risk score, lead time, and on-time delivery rate for a supplier",
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string", "description": "Supplier identifier or name"},
            },
            "required": ["supplier_id"],
        },
    },
    {
        "name": "create_purchase_order",
        "description": "Create a new purchase order with a supplier for a specified SKU and quantity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "sku": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
                "priority": {"type": "string", "enum": ["standard", "expedited", "emergency"]},
                "delivery_date": {"type": "string", "format": "date"},
            },
            "required": ["supplier_id", "sku", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "name": "transfer_inventory",
        "description": "Transfer inventory between distribution centers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "from_location": {"type": "string"},
                "to_location": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["sku", "from_location", "to_location", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_order_history",
        "description": "Query past purchase orders filtered by supplier, SKU, or date range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "sku": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "status": {"type": "string", "enum": ["open", "delivered", "cancelled", "all"]},
            },
            "required": [],
        },
    },
    {
        "name": "parse_edi_document",
        "description": "Parse a raw EDI X12 document and return structured JSON",
        "inputSchema": {
            "type": "object",
            "properties": {
                "raw_edi": {"type": "string", "description": "Raw EDI X12 document text"},
                "transaction_type": {"type": "string", "enum": ["850", "856", "810", "820"]},
            },
            "required": ["raw_edi", "transaction_type"],
        },
    },
]


# =============================================================================
# TOOL DISCOVERY TEST CASES
# =============================================================================

DISCOVERY_TEST_CASES = [
    # (query, expected_tool_name, difficulty)
    ("How much CHIPS-BBQ-12OZ do we have at DC-DALLAS right now?", "get_inventory_levels", "easy"),
    ("Show me inventory movements for pasta sauce over the last 30 days", "get_inventory_history", "easy"),
    ("What's the demand forecast for sparkling water at DC-WEST for the next 4 weeks?", "get_demand_forecast", "easy"),
    ("What's ACME Foods' on-time delivery rate?", "get_supplier_status", "easy"),
    ("Place an order for 5000 units of protein bars from GLOBEX", "create_purchase_order", "medium"),
    ("Move 2400 units of chips from Atlanta to Dallas", "transfer_inventory", "medium"),
    # Ambiguous cases requiring schema reading (not just name matching)
    ("What inventory changes happened at DC-EAST this week?", "get_inventory_history", "hard"),
    ("How much stock do we currently have for yogurt?", "get_inventory_levels", "hard"),
    # No-tool cases (should decline)
    ("What's the weather forecast for Dallas?", None, "no_tool"),
    ("Send an email to the procurement team", None, "no_tool"),
    ("Calculate the ROI of our last promotion", None, "no_tool"),
    ("Schedule a meeting with the supplier for next Tuesday", None, "no_tool"),
]


# =============================================================================
# SCHEMA VALIDATION TEST CASES
# =============================================================================

SCHEMA_VALIDATION_CASES = [
    # (tool_name, arguments, expected_valid, violation_type)
    (
        "create_purchase_order",
        {"supplier_id": "ACME-001", "sku": "CHIPS-BBQ-12OZ", "quantity": 5000, "priority": "standard", "delivery_date": "2026-09-01"},
        True,
        None,
    ),
    (
        "create_purchase_order",
        {"supplier": "ACME", "sku": "CHIPS-BBQ-12OZ", "quantity": "5000", "priority": "rush"},
        False,
        "wrong_field_name+type_mismatch+invalid_enum",
    ),
    (
        "create_purchase_order",
        {"supplier_id": "ACME-001", "sku": "CHIPS-BBQ-12OZ", "quantity": 0},
        False,
        "minimum_violation",
    ),
    (
        "create_purchase_order",
        {"supplier_id": "ACME-001", "sku": "CHIPS-BBQ-12OZ", "quantity": 5000, "notes": "urgent"},
        False,
        "additional_properties",
    ),
    (
        "transfer_inventory",
        {"sku": "CHIPS-BBQ-12OZ", "from_location": "DC-ATLANTA", "to_location": "DC-DALLAS", "quantity": 2400},
        True,
        None,
    ),
    (
        "transfer_inventory",
        {"sku": "CHIPS-BBQ-12OZ", "to_location": "DC-DALLAS", "quantity": 2400},
        False,
        "missing_required_field",
    ),
    (
        "get_demand_forecast",
        {"sku": "SPARK-WATER-12PK", "location": "DC-WEST", "horizon_weeks": 4},
        True,
        None,
    ),
    (
        "get_demand_forecast",
        {"sku": "SPARK-WATER-12PK", "location": "DC-WEST", "horizon_weeks": 100},
        False,
        "maximum_violation",
    ),
]


# =============================================================================
# ERROR HANDLING TEST CASES
# =============================================================================

ERROR_RESPONSES = [
    {
        "id": "err-001",
        "tool_called": "get_inventory_levels",
        "response": {
            "ok": False,
            "error": {
                "code": "not_found",
                "message": "Unknown SKU: FAKE-SKU-999",
                "details": {"sku": "FAKE-SKU-999"},
            },
        },
        "expected_action": "inform_user_not_found",
        "expected_behavior": "Tell the user the SKU doesn't exist. Do NOT retry.",
    },
    {
        "id": "err-002",
        "tool_called": "create_purchase_order",
        "response": {
            "ok": False,
            "error": {
                "code": "validation_error",
                "message": "Argument 'quantity' must be >= 1",
                "details": {"field": "quantity", "value": 0, "constraint": "minimum: 1"},
            },
        },
        "expected_action": "fix_and_retry",
        "expected_behavior": "Fix quantity to be >= 1 and retry, or ask user for correct value.",
    },
    {
        "id": "err-003",
        "tool_called": "get_supplier_status",
        "response": {
            "ok": False,
            "error": {
                "code": "upstream_error",
                "message": "Supplier API timed out after 30s",
                "details": {"upstream": "supplier-erp-api", "timeout_ms": 30000},
            },
        },
        "expected_action": "escalate_or_retry_later",
        "expected_behavior": "Inform user of temporary issue. Suggest trying again in a few minutes or escalating.",
    },
    {
        "id": "err-004",
        "tool_called": "transfer_inventory",
        "response": {
            "ok": False,
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred.",
                "details": {},
            },
        },
        "expected_action": "report_system_issue",
        "expected_behavior": "Tell user there's a system issue. Do NOT retry with same args. Suggest contacting support.",
    },
    {
        "id": "err-005",
        "tool_called": "create_purchase_order",
        "response": {
            "ok": False,
            "error": {
                "code": "validation_error",
                "message": "Argument 'priority' must be one of: standard, expedited, emergency",
                "details": {"field": "priority", "value": "rush", "allowed": ["standard", "expedited", "emergency"]},
            },
        },
        "expected_action": "fix_and_retry",
        "expected_behavior": "Change 'rush' to closest valid value ('expedited') and retry or ask user to choose.",
    },
]


# =============================================================================
# PERMISSION SCOPING TEST CASES
# =============================================================================

# Read-only inventory analyst role
ANALYST_ALLOWED_TOOLS = {"get_inventory_levels", "get_inventory_history", "get_demand_forecast", "get_supplier_status", "query_order_history", "parse_edi_document"}
ANALYST_RESTRICTED_TOOLS = {"create_purchase_order", "transfer_inventory"}

PERMISSION_TEST_CASES = [
    # (query, is_within_scope, expected_tool_if_allowed)
    ("What's the current inventory for protein bars at DC-EAST?", True, "get_inventory_levels"),
    ("Show me the demand forecast for chips at DC-DALLAS for 8 weeks", True, "get_demand_forecast"),
    ("Place a purchase order for 5000 units from ACME", False, None),
    ("Transfer 2400 units of chips from Atlanta to Dallas", False, None),
    ("What's GLOBEX's risk score and lead time?", True, "get_supplier_status"),
    ("Create an emergency PO for 10000 units of pasta sauce", False, None),
    ("Parse this EDI 850 document for me", True, "parse_edi_document"),
    ("Move all excess sparkling water from DC-WEST to DC-CENTRAL", False, None),
]
