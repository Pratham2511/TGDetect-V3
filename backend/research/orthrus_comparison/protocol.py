"""Paper split registration, identity policy, and conservative feature contract."""
from datetime import datetime
from zoneinfo import ZoneInfo

TRAIN_DAYS = (3,4,5,7,8,9,10)
VALIDATION_DAYS = (2,)
TEST_DAYS = (6,11,12,13)
# Official nanosecond conversion uses America/New_York, not UTC calendar dates.
TIMEZONE = 'America/New_York'


def partition(timestamp):
    dt = datetime.fromtimestamp(timestamp, ZoneInfo(TIMEZONE))
    if (dt.year, dt.month) != (2018,4):
        raise ValueError('Expected CADETS E3 April 2018 timestamps')
    for name, days in [('train',TRAIN_DAYS),('validation',VALIDATION_DAYS),('test',TEST_DAYS)]:
        if dt.day in days:
            return name
    raise ValueError('Timestamp outside paper split')


def require_coverage(days):
    required = set(TRAIN_DAYS + VALIDATION_DAYS + TEST_DAYS)
    missing = sorted(required-set(days))
    if missing:
        raise ValueError(f'Incomplete paper dataset: missing April days {missing}')


def identity(event_id, source, host, role, entity_id):
    # A missing ID is event-local. Do not create one global unknown entity.
    return (str(source), str(host), str(entity_id)) if entity_id is not None else (
        str(source), str(host), '@missing', str(event_id), str(role))


def reconstruction_query(src_attributes, dst_attributes):
    # Input includes endpoint attributes only. Current event type, action, raw
    # message, ground truth, rule ID and classification output never enter it.
    allowed = {'type','path','cmd_line','remote_ip','remote_port'}
    return {'type':None, 'attributes':{
        'src':{k:v for k,v in src_attributes.items() if k in allowed},
        'dst':{k:v for k,v in dst_attributes.items() if k in allowed}}}
