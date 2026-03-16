
import numpy
import re


def define_cocos(cocos_ind):
    """
    Returns dictionary with COCOS coefficients given a COCOS index

    https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit

    :param cocos_ind: COCOS index

    :return: dictionary with COCOS coefficients
    """

    cocos = dict.fromkeys(['sigma_Bp', 'sigma_RpZ', 'sigma_rhotp', 'sign_q_pos', 'sign_pprime_pos', 'exp_Bp'])

    # all multipliers shouldn't change input values if cocos_ind is None
    if cocos_ind is None:
        cocos['exp_Bp'] = 0
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = 0
        cocos['sign_pprime_pos'] = 0
        return cocos

    # if COCOS>=10, this should be 1
    cocos['exp_Bp'] = 0
    if cocos_ind >= 10:
        cocos['exp_Bp'] = +1

    if cocos_ind in [1, 11]:
        # These cocos are for
        # (1)  psitbx(various options), Toray-GA
        # (11) ITER, Boozer
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [2, 12, -12]:
        # These cocos are for
        # (2)  CHEASE, ONETWO, HintonHazeltine, LION, XTOR, MEUDAS, MARS, MARS-F
        # (12) GENE
        # (-12) ASTRA
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [3, 13]:
        # These cocos are for
        # (3) Freidberg*, CAXE and KINX*, GRAY, CQL3D^, CarMa, EFIT* with : ORB5, GBSwith : GT5D
        # (13)  CLISTE, EQUAL, GEC, HELENA, EU ITM-TF up to end of 2011
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [4, 14]:
        # These cocos are for
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [5, 15]:
        # These cocos are for
        # (5) TORBEAM, GENRAY^
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [6, 16]:
        # These cocos are for
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [7, 17]:
        # These cocos are for
        # (17) LIUQE*, psitbx(TCV standard output)
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [8, 18]:
        # These cocos are for
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = +1

    return cocos


def cocos_transform(cocosin_index, cocosout_index):
    """
    Returns a dictionary with coefficients for how various quantities should get multiplied in order to go from cocosin_index to cocosout_index

    https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit

    :param cocosin_index: COCOS index in

    :param cocosout_index: COCOS index out

    :return: dictionary with transformation multipliers
    """

    # Don't transform if either cocos is undefined
    if (cocosin_index is None) or (cocosout_index is None):
        # printd("No COCOS tranformation for " + str(cocosin_index) + " to " + str(cocosout_index), topic='cocos')
        sigma_Ip_eff = 1
        sigma_B0_eff = 1
        sigma_Bp_eff = 1
        exp_Bp_eff = 0
        sigma_rhotp_eff = 1
    else:
        # printd("COCOS tranformation from " + str(cocosin_index) + " to " + str(cocosout_index), topic='cocos')
        cocosin = define_cocos(cocosin_index)
        cocosout = define_cocos(cocosout_index)

        sigma_Ip_eff = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_B0_eff = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_Bp_eff = cocosin['sigma_Bp'] * cocosout['sigma_Bp']
        exp_Bp_eff = cocosout['exp_Bp'] - cocosin['exp_Bp']
        sigma_rhotp_eff = cocosin['sigma_rhotp'] * cocosout['sigma_rhotp']

    # Transform
    transforms = {}
    transforms['1/PSI'] = sigma_Ip_eff * sigma_Bp_eff / (2 * numpy.pi) ** exp_Bp_eff
    transforms['invPSI'] = transforms['1/PSI']
    transforms['dPSI'] = transforms['1/PSI']
    transforms['F_FPRIME'] = transforms['dPSI']
    transforms['PPRIME'] = transforms['dPSI']
    transforms['PSI'] = sigma_Ip_eff * sigma_Bp_eff * (2 * numpy.pi) ** exp_Bp_eff
    transforms['Q'] = sigma_Ip_eff * sigma_B0_eff * sigma_rhotp_eff
    transforms['TOR'] = sigma_B0_eff
    transforms['BT'] = transforms['TOR']
    transforms['IP'] = transforms['TOR']
    transforms['F'] = transforms['TOR']
    transforms['POL'] = sigma_B0_eff * sigma_rhotp_eff
    transforms['BP'] = transforms['POL']
    transforms[None] = 1

    # printd(transforms, topic='cocos')

    return transforms


def compare_version(version1, version2):
    """
    Compares two version numbers and determines which one, if any, is greater.

    This function can handle wildcards (eg. 1.1.*)
    Most non-numeric characters are removed, but some are given special treatment.
    a, b, c represent alpha, beta, and candidate versions and are replaced by numbers -3, -2, -1.
        So 4.0.1-a turns into 4.0.1.-3, 4.0.1-b turns into 4.0.1.-2, and then -3 < -2
        so the beta will be recognized as newer than the alpha version.
    rc# is recognized as a release candidate that is older than the version without the rc
        So 4.0.1_rc1 turns into 4.0.1.-1.1 which is older than 4.0.1 because 4.0.1 implies 4.0.1.0.0.
        Also 4.0.1_rc2 is newer than 4.0.1_rc1.

    :param version1: str
        First version to compare

    :param version2: str
        Second version to compare

    :return: int
        1 if version1 > version2
        -1 if version1 < version2
        0 if version1 == version2
        0 if wildcards allow version ranges to overlay. E.g. 4.* vs. 4.1.5 returns 0 (equal)
    """
    version1 = sanitize_version_number(version1)
    version2 = sanitize_version_number(version2)

    # Handle version wildcards
    if '*' in version1 or '*' in version2:
        version1 = version1.split('.')
        version2 = version2.split('.')
        start_asterix = False
        for k in range(max([len(version1), len(version2)])):
            if (k < len(version1) and version1[k] == '*') or (k < len(version2) and version2[k] == '*'):
                start_asterix = True
            if start_asterix:
                if k < len(version1):
                    version1[k] = '*'
                else:
                    version1.append('*')
                if k < len(version2):
                    version2[k] = '*'
                else:
                    version2.append('*')
        version1 = '.'.join(version1)
        version2 = '.'.join(version2)

    def version_int(x):
        if x in ['', ' ']:
            return 0
        elif x in '*':
            return x
        else:
            return int(x)

    def normalize(v):
        return [version_int(x) for x in re.sub(r'(\.0+)*$', '', v).split('.')]

    n1 = normalize(version1)
    n2 = normalize(version2)
    dn1 = len(n1) - len(n2)
    if dn1 < 0:
        n1 += [0] * -dn1
    elif dn1 > 0:
        n2 += [0] * dn1
    return (n1 > n2) - (n1 < n2)


def sanitize_version_number(version):
    """Removes common non-numerical characters from version numbers obtained from git tags, such as '_rc', etc."""
    if version.startswith('.'):
        version = '-1' + version
    # Replace alpha, beta, release candidate *-a, *-b *-c endings with -3, -2, -1
    version = re.sub(r'([0-9]+)-?c', r'\1.-1', version)
    version = re.sub(r'([0-9]+)-?b', r'\1.-2', version)
    version = re.sub(r'([0-9]+)-?a', r'\1.-3', version)
    # More release candidate things
    version = re.sub(r'([0-9\-]+)_?rc([0-9\-]+)', r'\1\.-1\.\2', version)
    # Get rid of remaining non-numerics except for .-*
    version = re.sub(r'[^0-9\.\*\-]', '.', version)
    # Remove any -[char]
    version = re.sub(r'-[a-zA-Z\.]', '.', version)
    # Suppress repeated '.'
    while '..' in version:
        version = version.replace('..', '.')
    return version


def tolist(data, empty_lists=None):
    """
    makes sure that the returned item is in the format of a list

    :param data: input data

    :param empty_lists: list of values that if found will be filtered out from the returned list

    :return: list format of the input data
    """
    import numpy as np

    data = evalExpr(data)
    if isinstance(data, str):
        data = [data]
    if empty_lists:
        if not np.iterable(data):
            if data in empty_lists:
                return []
        else:
            data0 = data
            data = []
            for k in data0:
                if k not in empty_lists:
                    data.append(k)
    if isinstance(data, np.ndarray):
        return np.atleast_1d(data).tolist()
    elif isinstance(data, dict):
        return [data]
    try:
        return list(data)
    except TypeError:
        return [data]


# ---------------------
# evaluate expressions
# ---------------------
def isinstance_str(inv, cls):
    """
    checks if an object is of a certain type by looking at the class name (not the class object)
    This is useful to circumvent the need to load import Python modules.

    :param inv: object of which to check the class

    :param cls: string or list of string with the name of the class(es) to be checked

    :return: True/False
    """
    if isinstance(cls, str):
        cls = [cls]
    if hasattr(inv, '__class__') and hasattr(inv.__class__, '__name__') and inv.__class__.__name__ in cls:
        return True
    return False


def evalExpr(inv):
    """
    Return the object that dynamic expressions return when evaluated
    This allows OMFITexpression('None') is None to work as one would expect.
    Epxressions that are invalid they will raise an OMFITexception when evaluated

    :param inv: input object

    :return:

        * If inv was a dynamic expression, returns the object that dynamic expressions return when evaluated

        * Else returns the input object

    """
    if isinstance_str(inv, 'OMFITexpressionError'):
        raise ValueError('Invalid expression')
    elif isinstance_str(inv, ['OMFITexpression', 'OMFITiterableExpression']) and hasattr(inv, '_value_'):
        tmp = inv._value_()
        if isinstance_str(tmp, 'OMFITexpressionError'):
            raise ValueError('Invalid expression:\n' + inv.error)
        return tmp
    else:
        return inv


def freezeExpr(me, remove_OMFITexpressionError=False):
    """
    Traverse a dictionary and evaluate OMFIT dynamic expressions in it
    NOTE: This function operates in place

    :param me: input dictionary

    :param remove_OMFITexpressionError: remove entries that evaluate as OMFITexpressionError

    :return: updated dictionary
    """
    for kid in list(me.keys()):
        if isinstance_str(me[kid], ['OMFITexpression', 'OMFITiterableExpression']):
            try:
                me[kid] = evalExpr(me[kid])
            except Exception:
                del me[kid]
                continue
        elif isinstance_str(me[kid], 'OMFITexpressionError'):
            if remove_OMFITexpressionError:
                del me[kid]
                continue
            else:
                raise ValueError('Invalid expression:\n' + me[kid].error)
        if isinstance(me[kid], dict):
            freezeExpr(me[kid], remove_OMFITexpressionError=remove_OMFITexpressionError)
# ---------------------
# checktypes
# ---------------------
def is_none(inv):
    """
    This is a convenience function to evaluate if a object or an expression is None
    Use of this function is preferred over testing if an expression is None
    by using the == function. This is because np arrays evaluate == on a per item base

    :param inv: input object

    :return: True/False
    """
    if inv is None:
        return True
    elif isinstance_str(inv, ['OMFITexpression', 'OMFITiterableExpression']):
        return evalExpr(inv) is None
    return False


def is_bool(value):
    """
    Convenience function check if value is boolean

    :param value: value to check

    :return: True/False
    """
    return value in [True, False]


def is_int(value):
    """
    Convenience function check if value is integer

    :param value: value to check

    :return: True/False
    """
    import numpy as np

    return isinstance(value, (int, np.integer))


def is_float(value):
    """
    Convenience function check if value is float

    :param value: value to check

    :return: True/False
    """
    import numpy as np

    return isinstance(value, (float, np.floating))


def is_numeric(value):
    """
    Convenience function check if value is numeric

    :param value: value to check

    :return: True/False
    """
    try:
        0 + value
        return True
    except TypeError:
        return False


def is_number_string(my_string):
    """
    Determines whether a string may be parsed as a number
    :param my_string: string
    :return: bool
    """
    try:
        float(my_string)
    except ValueError:
        return False
    else:
        return True


def is_alphanumeric(value):
    """
    Convenience function check if value is alphanumeric

    :param value: value to check

    :return: True/False
    """
    if isinstance(value, str):
        return True
    try:
        0 + value
        return True
    except TypeError:
        return False


def is_array(value):
    """
    Convenience function check if value is list/tuple/array

    :param value: value to check

    :return: True/False
    """
    import numpy as np

    return isinstance(value, (list, tuple, np.ndarray))


def is_string(value):
    """
    Convenience function check if value is string

    :param value: value to check

    :return: True/False
    """
    return isinstance(value, str)


def is_email(value):
    if isinstance(value, str):
        return re.findall('i?[\w\-\.]+@[\w\-\.]+\.[\w\-\.]+', value)


def is_int_array(val):
    """
    Convenience function check if value is a list/tuple/array of integers

    :param value: value to check

    :return: True/False
    """
    import numpy as np

    if is_array(val):
        try:
            tmp = np.atleast_1d(val).astype(int)
        except TypeError:
            return False
        if np.all(np.atleast_1d(val) == tmp):
            return True
    return False

