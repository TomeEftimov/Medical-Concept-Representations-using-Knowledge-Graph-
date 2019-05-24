def cmp_to_key(mycmp):
    
    'Convert a cmp= function into a key= function'
    
    class K(object):
        def __init__(self, obj, *args):
        self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0  
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    
    return K

def compare_datetime(d1, d2):
    date1, time1 = d1.split(' ')
    year1, month1, day1 = map(int,date1.split('-'))
    hour1, minutes1, sec1 = map(int,time1.split(':'))
    date2, time2 = d2.split(' ')
    year2, month2, day2 = map(int,date2.split('-'))
    hour2, minutes2, sec2 = map(int,time2.split(':'))
    if year1 > year2:
        return 1
    elif year1 < year2:
        return -1
    elif month1 > month2:
        return 1
    elif month1 < month2:
        return -1
    elif day1 > day2:
        return 1
    elif day1 < day2:
        return -1
    elif hour1 > hour2:
        return -1
    elif hour1 < hour2:
        return -1
    elif minutes1 > minutes2:
        return 1
    elif minutes1 < minutes2:
        return -1
    elif sec1 >= sec2:
        return 1
    else:
        return 0