"""Travel-booking agent under test — vendored from the Chatbot module.

This is the SAME workload as ../Chatbot (six tools, mock data, confirm-before-book
system prompt), vendored here so the Model Upgrade module is self-contained. The
one difference: the agent's model call is routed through the single AgentCore
Gateway inference seam, so switching between the SOURCE (GPT-4.1 via OpenAI) and
TARGET (Sonnet 5.5 via Bedrock) models is only a change of the `model` string —
the agent code below never changes.

The system prompt here is the SEED prompt the hill climb starts from. The optimizer
proposes mutations of SYSTEM_PROMPT; everything else (tools, mock data) is fixed so
score movement reflects prompt + model, not external variability.
"""

from __future__ import annotations

from datetime import date

from strands import Agent, tool

# --- Booking state (in-memory, reset between scenarios) ---------------------
bookings: dict = {}
booking_counter = 0


def reset_state() -> None:
    """Reset booking state so scenarios do not contaminate each other."""
    global bookings, booking_counter
    bookings = {}
    booking_counter = 0


# --- Tools (identical to the Chatbot module) --------------------------------
@tool
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> str:
    """Search for available flights between two cities.

    Args:
        origin: Departure city or airport code (e.g. 'New York' or 'JFK')
        destination: Arrival city or airport code (e.g. 'London' or 'LHR')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date in YYYY-MM-DD format for round trips
    """
    flights = [
        {"flight": "AA101", "airline": "American Airlines", "departs": "08:00", "arrives": "20:00", "price": 450, "class": "Economy"},
        {"flight": "BA202", "airline": "British Airways", "departs": "11:30", "arrives": "23:45", "price": 620, "class": "Economy"},
        {"flight": "UA303", "airline": "United Airlines", "departs": "14:00", "arrives": "02:15", "price": 890, "class": "Business"},
    ]
    trip = f"round-trip (return: {return_date})" if return_date else "one-way"
    lines = [f"Flights from {origin} to {destination} on {departure_date} ({trip}):"]
    for f in flights:
        lines.append(f" {f['flight']} | {f['airline']} | {f['departs']}-{f['arrives']} | ${f['price']} | {f['class']}")
    return "\n".join(lines)


@tool
def search_hotels(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search for available hotels in a city.

    Args:
        city: Destination city
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests (default 1)
    """
    hotels = [
        {"name": "Grand Plaza Hotel", "stars": 5, "price_per_night": 320, "amenities": "Pool, Spa, Restaurant"},
        {"name": "City Center Inn", "stars": 3, "price_per_night": 95, "amenities": "Free WiFi, Breakfast"},
        {"name": "Boutique Stay", "stars": 4, "price_per_night": 180, "amenities": "Gym, Bar, Concierge"},
    ]
    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    lines = [f"Hotels in {city} | {check_in} to {check_out} ({nights} nights, {guests} guest(s)):"]
    for h in hotels:
        total = h["price_per_night"] * nights
        lines.append(f" {h['name']} ({h['stars']}*) | ${h['price_per_night']}/night (${total} total) | {h['amenities']}")
    return "\n".join(lines)


@tool
def book_flight(flight_number: str, passenger_name: str, origin: str, destination: str, travel_date: str) -> str:
    """Book a flight for a passenger.

    Args:
        flight_number: Flight number to book (e.g. 'AA101')
        passenger_name: Full name of the passenger
        origin: Departure city or airport
        destination: Arrival city or airport
        travel_date: Travel date in YYYY-MM-DD format
    """
    global booking_counter
    booking_counter += 1
    ref = f"FLT-{booking_counter:04d}"
    bookings[ref] = {"type": "flight", "flight": flight_number, "passenger": passenger_name,
                     "route": f"{origin} -> {destination}", "date": travel_date, "status": "Confirmed"}
    return f"Flight booked! Ref: {ref} | {flight_number} | {origin} -> {destination} on {travel_date} | Passenger: {passenger_name}"


@tool
def book_hotel(hotel_name: str, guest_name: str, city: str, check_in: str, check_out: str) -> str:
    """Book a hotel room for a guest.

    Args:
        hotel_name: Name of the hotel to book
        guest_name: Full name of the guest
        city: City where the hotel is located
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
    """
    global booking_counter
    booking_counter += 1
    ref = f"HTL-{booking_counter:04d}"
    bookings[ref] = {"type": "hotel", "hotel": hotel_name, "guest": guest_name,
                     "city": city, "check_in": check_in, "check_out": check_out, "status": "Confirmed"}
    return f"Hotel booked! Ref: {ref} | {hotel_name} in {city} | {check_in} to {check_out} | Guest: {guest_name}"


@tool
def get_bookings() -> str:
    """Retrieve all current bookings."""
    if not bookings:
        return "No bookings found."
    lines = ["Current bookings:"]
    for ref, b in bookings.items():
        if b["type"] == "flight":
            lines.append(f" [{ref}] Flight {b['flight']} | {b['route']} on {b['date']} | {b['passenger']} | {b['status']}")
        else:
            lines.append(f" [{ref}] Hotel {b['hotel']} in {b['city']} | {b['check_in']} to {b['check_out']} | {b['guest']} | {b['status']}")
    return "\n".join(lines)


@tool
def cancel_booking(booking_ref: str) -> str:
    """Cancel an existing booking.

    Args:
        booking_ref: Booking reference number (e.g. 'FLT-0001' or 'HTL-0002')
    """
    if booking_ref not in bookings:
        return f"Booking {booking_ref} not found."
    bookings[booking_ref]["status"] = "Cancelled"
    return f"Booking {booking_ref} has been cancelled."


ALL_TOOLS = [search_flights, search_hotels, book_flight, book_hotel, get_bookings, cancel_booking]

# The SEED system prompt — the hill climb starts here and mutates THIS string only.
SYSTEM_PROMPT = (
    "You are a helpful travel booking assistant. You help users search for flights and hotels, "
    "make bookings, view existing reservations, and cancel bookings. "
    "Always confirm details with the user before completing a booking. "
    "Use today's date as context when dates are not fully specified."
)


def build_agent(system_prompt: str, model, session_id: str) -> Agent:
    """Build the travel agent with a given prompt and a Gateway-routed model.

    `model` is whatever Strands model object the notebook built against the single
    Gateway inference endpoint (see notebook Section 2). The SAME agent code serves
    the source and target models — only the model object's `model` string differs.
    """
    return Agent(
        system_prompt=system_prompt,
        tools=ALL_TOOLS,
        model=model,
        trace_attributes={
            "gen_ai.conversation.id": session_id,
            "session.id": session_id,
        },
        callback_handler=None,
    )
