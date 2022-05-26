def greater_zero_rule(field_name: str):
    return (lambda x: x > 0, f'{field_name} should be positive')


def non_negative_rule(field_name: str):
    return (lambda x: x >= 0, f'{field_name} should be non-negative')


def between_zero_one_rule(field_name: str):
    return (lambda x: x >= 0 and x <= 1, f'{field_name} should be between 0 (included) and 1 (included)')
