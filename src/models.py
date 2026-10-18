"""Milestone 3 (identity verification) -- structured-extraction schema.

The verify node does NOT let the LLM free-type an identifier. It binds this
SHAPE to the model (llm.with_structured_output(UserInput)) so the model must
return a filled-in object with exactly these fields -- or all-null if the user
gave no identifier. This is the inverse of tool-calling: tool-calling binds
functions the model may CALL; structured output binds a schema the model must
FILL. Pydantic v2 validates the result before any DB lookup runs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """One customer identifier extracted from the user's own message.

    All three are optional because a customer may provide any one of them (or,
    on an off-topic / anonymous message, none). The verify node tries them in
    order (id -> email -> phone) and confirms each against the database before
    trusting it.
    """

    customer_id: int | None = Field(
        default=None,
        description="The numeric customer ID, if the user explicitly stated one.",
    )
    email: str | None = Field(
        default=None,
        description="The customer's email address, if the user stated one.",
    )
    phone: str | None = Field(
        default=None,
        description="The customer's phone number, if the user stated one.",
    )

    def is_empty(self) -> bool:
        """True when the user supplied no identifier at all."""
        return self.customer_id is None and not self.email and not self.phone



class UserProfile(BaseModel):
    """A customer's remembered music preferences, persisted across conversations.

    This is the shape STORED in the memory store (one per customer). It only ever
    grows: new preferences are set-union merged in, existing ones are never removed.
    """

    music_preferences: list[str] = Field(
        default_factory=list,
        description="Lowercase preference tags the customer has stated (genres, artists, styles).",
    )


class PreferenceExtraction(BaseModel):
    """What the memory extractor pulls from ONE message.

    Bound to the LLM via with_structured_output. The model fills `preferences` with
    tastes the customer EXPLICITLY stated this turn, or leaves it empty when the
    message is a question/request (which must save nothing).
    """

    preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Music tastes explicitly stated in this message (e.g. 'I love jazz' -> "
            "['jazz']). EMPTY for a question or request (e.g. 'do you have jazz?')."
        ),
    )