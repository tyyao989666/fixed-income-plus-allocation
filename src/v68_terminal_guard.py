"""Conservative dated terminal guard; separate signal eligibility and cash timing."""
import pandas as pd


def terminal_guard(schedule, decision_date, held_quantity=0., confirmed_result=None):
    """Schedule is reviewed evidence; later result never backdates cash certainty.

    Buy/exit flags are policy recommendations, not proof an order was filled.
    Receivable valuation remains separate from historical model eligibility.
    """
    day = pd.Timestamp(decision_date).normalize()
    available = pd.Timestamp(schedule['conservative_available_after_close'])
    record = pd.Timestamp(schedule['record_date'])
    redeem = pd.Timestamp(schedule['redemption_date'])
    pay = pd.Timestamp(schedule['payment_date'])
    amount = float(schedule['gross_cash_yuan_per_bond'])
    if available > record or record >= redeem or redeem > pay or amount <= 0 or held_quantity < 0:
        raise ValueError('Invalid reviewed terminal schedule')
    # Conservative availability is after the named close, so same day is excluded.
    known = day > available
    state = dict(schedule_known=known, buy_allowed=True, exit_required=False,
        claim_expected=False, usable_cash_confirmed=False, expected_cash=0.,
        cash_receipt=0., action='no_known_terminal_override')
    if not known:
        return state
    state['buy_allowed'] = False
    if day <= record:
        state.update(exit_required=held_quantity > 0, action='exit_if_executable_before_deadline')
    else:
        state.update(claim_expected=held_quantity > 0, expected_cash=held_quantity*amount,
                     action='nontradable_claim_not_free_cash')
    if confirmed_result is not None:
        published = pd.Timestamp(confirmed_result['result_publication_date'])
        confirmed_pay = pd.Timestamp(confirmed_result['payment_date'])
        if confirmed_result['ts_code'] != schedule['ts_code'] or confirmed_pay != pay:
            raise ValueError('Settlement identity mismatch')
        if day > published and day >= pay:
            state.update(usable_cash_confirmed=True, action='reconcile_payment_and_cancel_claim')
    # Idempotent per-event posting belongs in a quantity ledger. This pure function
    # deliberately never emits a repeatable cash receipt on each subsequent day.
    return state
