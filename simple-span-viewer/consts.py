from colorama import Back

COLOUR_MAPPING = {
    # for claims
    'claim': Back.RED,
    'per_exp': Back.BLUE,
    'claim_per_exp': Back.WHITE,
    'question': Back.GREEN,
    # for PICO
    'population': Back.RED,
    'intervention': Back.BLUE,
    'outcome': Back.WHITE
}

SUBREDDIT_ID_TO_POPULATION = {
    "t5_2rtve": "lupus",
    "t5_2syer": "gout",
    "t5_2s3g1": "ibs",
    "t5_2tyg2": "Psychosis",
    "t5_395ja": "costochondritis",
    "t5_2saq9": "POTS",
    "t5_2s23e": "MultipleSclerosis",
    "t5_2s1h9": "Epilepsy",
    "t5_2qlaa": "GERD",
    "t5_2r876": "CysticFibrosis",
}

TARGET_CLASSES = [
    'claim',
    'per_exp',
    'claim_per_exp',
    'question'
]
