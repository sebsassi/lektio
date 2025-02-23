import numpy as np
import time
import datetime
import re
import typing


_epochs = {'CE': 0, 'J2000': 730485.5, 'UNIX': 719528.0}

def is_leap_year(year: int) -> bool:
    """
    Check whether year is a leap year.

    Parameters
    ----------
    year : int

    Returns
    -------
    bool
    """
    return (year%4 == 0) and (year%100 != 0) or (year%400 == 0)


def str_date_to_struct_time(date: str) -> time.struct_time:
    """
    Turn a date string into a time.struct_time object.

    Parameters
    ----------
    date : str
        Input date string.

    Returns
    -------
    time.struct_time
        Output struct_time.
    """
    num_fields = len(re.split(r'\W+', date))
    date_format = '-'.join(("%Y", "%m", "%d", "%H", "%M", "%S")[:num_fields])
    return time.strptime(date, date_format)


def date_to_struct_time(date: time.struct_time | str) -> time.struct_time:
    if isinstance(date, time.struct_time):
        return date
    elif isinstance(date, str):
        return str_date_to_struct_time(date)
    else:
        raise TypeError("Argument must be a string or time.time_struct.")


def _days_in_years(year: int) -> int:
    """
    Compute the number of days between parameter year and the year 0.

    The number of days is measured in the Gregorian calendar regardless
    of input year.

    Parameters
    ----------
    year : int

    Returns
    -------
    int
    """
    return 365*year + np.ceil(year/4) - np.ceil(year/100) + np.ceil(year/400)


def struct_time_to_time(date: time.struct_time, epoch: str = "J2000") -> float:
    """Convert a struct_time to time in days since epoch."""
    full_days = _days_in_years(date.tm_year) + date.tm_yday - 1
    fractional_day = (date.tm_hour/24 + date.tm_min/1440 + date.tm_sec/86400)
    return full_days + fractional_day - _epochs[epoch]


def str_date_to_time(date: str, epoch: str = "J2000") -> float:
    """
    Convert a date string to mean sloar days since epoch in UTC.

    Parameters
    ----------
    date : string
        String representation of date. Accepted formats are
        'YYYY-MM-DD hh:mm:ss' or 'YYYY-MM-DD'.
    epoch : {'CE', 'J2000',}
        Epoch defining t = 0. 'CE' is 0000-00-00 00:00:00 UTC,
        'J2000' is 2000-01-01 12:00:00 UTC.

    Returns
    -------
    float
        Time since epoch in mean solar days.
    """
    return struct_time_to_time(str_date_to_struct_time(date), epoch)


def time_to_struct_date(t: float, epoch: str = "J2000") -> time.struct_time:
    """Convert a time in days since epoch into time.struct_time object."""
    return time.gmtime((t + _epochs[epoch] - _epochs['UNIX'])*86400)


def time_to_date(t: float, epoch: str = "J2000") -> str:
    """
    Convert a time in mean solar days since given epoch to a date.

    Inverse of str_date_to_time.

    Parameters
    ----------
    time : float
        Time in mean solar days since epoch
    epoch : {'CE', 'J2000'}
        Epoch defining t = 0. 'CE' is 0000-00-00 00:00:00 UTC,
        'J2000' is 2000-01-01 12:00:00 UTC.

    Returns
    -------
    str
        String representation of the date.
    """
    return time.strftime(
            "%Y-%m-%d %H:%M:%S", time_to_struct_date(t, epoch='J2000'))


class TimeInterval(np.ndarray):
    """
    Representation of a time interval.

    This object represents a time interval defined by a start date,
    an end date, and a number of points in the interval. This is a
    subclass of numpy.ndarray. The purpose is that it can be used in
    compuations involving points in time, but with easy initialization
    from dates (e.g. TimeInterval('2002-02-03', '2002-02-04', 5)) and
    access to the dates corresponding to the times.

    The numbers in the array are days relative to a specific epoch,
    which is set by the epoch keyword.

    In ufuncs the TimeInterval just acts as a regular numpy.ndarray.
    Namely, ufuncs just output instances of numpy.ndarray.

    Attributes
    ----------
    epoch : {'CE', 'J2000', 'UNIX'}
        Epoch defining t = 0.
    start_date : time.struct_time
        Start date.
    end_date : time.struct_time
        End date.
    """
    def __new__(
        cls, start_date: time.struct_time | str,
        end_date: typing.Optional[time.struct_time | str] = None,
        size: int = 1, epoch: str = "J2000"
    ):
        """
        Parameters
        ----------
        start_date : time.struct_time, str
            Start date of the interval.
        end_date : time.struct_time, str, optional
            End date of the interval. If not provided will be same as
            start_date
        size : int, optional
            Number of points in the time interval.
        epoch : {'CE', 'J2000', 'UNIX'}, optional
            Epoch defining t = 0.

        Notes
        -----
        The TimeInterval object creation from start_date, end_date,
        and size is similar to an array creation using numpy.linspace.
        """
        end_date = start_date if end_date is None else end_date
        _start_date = date_to_struct_time(start_date)
        _end_date = date_to_struct_time(end_date)
        if size < 1:
            raise ValueError("size must be greater than zero.")
        st = struct_time_to_time(_start_date, epoch)
        et = struct_time_to_time(_end_date, epoch)
        obj = np.linspace(st, et, size).view(cls)
        if epoch not in _epochs.keys():
            return ValueError(
                f"Unknown epoch {epoch}. Epoch should be one of "
                f"{set(_epochs)}")
        obj.epoch = epoch
        obj._start_date = _start_date
        obj._end_date = _end_date
        return obj


    def __array_finalize__(self, obj):
        if obj is None: return None
        self.epoch = getattr(obj, 'epoch', 'J2000')
        self._start_date = time_to_struct_date(self.flat[0], self.epoch)
        self._end_date = time_to_struct_date(self.flat[-1], self.epoch)


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # All ufunc operations on a TimeInterval object just reduce the ouput
        # to a plain ndarray
        as_ndarray = (lambda x:
                x.view(np.ndarray) if isinstance(x, TimeInterval) else x)
        in_ = [as_ndarray(input_) for input_ in inputs]

        return super().__array_ufunc__(ufunc, method, *in_, **kwargs)


    def str_start_date(self, short: bool = False) -> str:
        """
        String representation of the start date.

        Parameters
        ----------
        short : bool, optional
            If true, don't include sub-day information.

        Returns
        -------
        str
            Start date as string.
        """
        return time.strftime(
                "%Y-%m-%d" if short else "%Y-%m-%d-%H-%M-%S", self._start_date)


    def str_end_date(self, short: bool = False) -> str:
        """
        String representation of the end date.

        Parameters
        ----------
        short : bool
            If true, don't include sub-day information.

        Returns
        -------
        str
            End date as string.
        """
        return time.strftime(
                "%Y-%m-%d" if short else "%Y-%m-%d-%H-%M-%S", self._end_date)


    def id_string(self, sep: str = "_") -> str:
        """
        Human readable string identifying the time interval.

        The string contains the relevant data that define the time
        interval: start date, end date, and number of points. This
        string is NOT guaranteed to be unique.

        Parameters
        ----------
        sep : str, optional
            Separator between elements of the string.

        Returns
        -------
        str
            String representation of the interval.
        """
        return sep.join((
                self.str_start_date(), self.str_end_date(), str(self.size)))
