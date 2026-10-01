"""Mortgage calculator."""

from dataclasses import dataclass, asdict

DEFAULT_DOWN_PAYMENT_PERCENT = 20.0
DEFAULT_INTEREST_RATE = 6.5
DEFAULT_YEARS = 30


@dataclass
class MortgageResult:
    price: float
    down_payment: float
    loan_amount: float
    monthly_payment: float
    total_interest: float
    total_cost: float
    interest_rate: float
    years: int

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def mortgage(
    price: float,
    down_payment_percent: float = DEFAULT_DOWN_PAYMENT_PERCENT,
    interest_rate: float = DEFAULT_INTEREST_RATE,
    years: int = DEFAULT_YEARS,
) -> MortgageResult:
    """Standard fixed-rate amortization: M = P·r(1+r)^n / ((1+r)^n − 1)."""
    if price <= 0:
        raise ValueError("price must be positive")
    if not 0 <= down_payment_percent <= 100:
        raise ValueError("down payment must be between 0 and 100%")
    if interest_rate < 0:
        raise ValueError("interest rate cannot be negative")
    if years <= 0:
        raise ValueError("loan term must be positive")

    down = price * down_payment_percent / 100
    loan = price - down
    r = interest_rate / 100 / 12
    n = years * 12
    monthly = loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_paid = monthly * n

    return MortgageResult(
        price=price,
        down_payment=down,
        loan_amount=loan,
        monthly_payment=monthly,
        total_interest=total_paid - loan,
        total_cost=total_paid + down,
        interest_rate=interest_rate,
        years=years,
    )
