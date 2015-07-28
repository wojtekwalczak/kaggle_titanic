from __future__ import print_function

import numpy as np
from titanic.db.loader import data

print(data.groupby('Survived').count())

