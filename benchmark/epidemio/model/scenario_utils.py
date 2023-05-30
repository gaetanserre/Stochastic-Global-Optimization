"""Definition of the scenario class."""

import numpy as np
from datetime import datetime, date, timedelta


class Scenario:
    def __init__(self, date_start, t_end, cf_evolution, date_format):
        """
        Args:
            args: Arguments parsed in execute.py
        """
        self.date_start = date_start
        self.date_format = date_format
        self.t_end = t_end

        # Creating lists
        dates = [0]
        cf_list = [1.]
        for date in cf_evolution.keys():
            dates.append(self.date_to_id(date))
            cf_list.append(cf_evolution[date])

        # Ordering lists and converting them to arrays
        arg_order = np.argsort(dates)
        dates = np.array(dates)[arg_order]
        cf_list = np.array(cf_list)[arg_order]

        self.contact_factor_evolution = {'dates': dates, 'contact_factor': cf_list}

    def date_to_id(self, date_d):
        """Converts a date to the corresponding ID."""
        if isinstance(date_d, date):
            end = date_d
        else:
            end = datetime.strptime(date_d, self.date_format)
        if type(self.date_start) is str:
            begin = datetime.strptime(self.date_start, self.date_format)
        else:
            begin = self.date_start
        return (end - begin).days

    def id_to_date(self, id):
        """Converts an ID to the corresponding date."""
        return self.date_start + timedelta(days=id)
