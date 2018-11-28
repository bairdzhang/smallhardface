import requests
import json
import time
import collections
import base64
import signal
from easydict import EasyDict
import logging
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


class Timeout():
    """Timeout class using ALARM signal."""

    def __init__(self, sec=10, k=5):
        self.sec = sec
        self.last_calls = collections.deque(k * [True], k)

    def set_timer(self, sec):
        self.sec = sec

    def anysuccess(self):
        return any(self.last_calls)

    def __enter__(self):
        self.last_calls.appendleft(True)
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        self.last_calls[0] = False
        raise TimeoutError()


TO = Timeout(10)


def grace(f):
    def wrapper(*args, **kwargs):
        try:
            with TO:
                return f(*args, **kwargs)
        except TimeoutError:
            logger.warning("TB timeout")
            if not TO.anysuccess():
                logger.warning("TB disabled due to continuous timeout")
                TO.set_timer(0.001)
        except:
            logger.warning("TB function error")

    return wrapper


try:
    # Python 2
    from urllib import quote_plus
except ImportError:
    # Python 3
    from urllib.parse import quote_plus

try:
    basestring
except NameError:
    basestring = str


class Fake(object):
    def __getattribute__(self, attr):
        def fake_attr(*args, **kwargs):
            pass

        return fake_attr


tb = EasyDict()

tb.client = Fake()
tb.sess = Fake()


class Tensorboard(object):
    @grace
    def __init__(self, hostname="localhost", port=8889):
        self.hostname = hostname
        self.port = port
        self.url = self.hostname + ":" + str(self.port)
        # TODO use urlparse
        if not (self.url.startswith("http://")
                or self.url.startswith("https://")):
            self.url = "http://" + self.url

        # check server is working (not only up).
        try:
            r = requests.get(self.url)
            # HACK remove this assertion to make program running
            # if not r.ok:
            #     raise RuntimeError("Something went wrong!" +
            #                        " Server sent: {}.".format(r.text))

        except requests.ConnectionError:
            msg = "The server at {}:{} does not appear to be up!"
            raise ValueError(msg.format(self.hostname, self.port))

    @grace
    def get_experiment_names(self):
        query = "/data"
        r = requests.get(self.url + query)
        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))
        else:
            experiments = json.loads(r.text)
        return experiments

    @grace
    def open_experiment(self, xp_name):
        assert (isinstance(xp_name, basestring))
        return TBExp(xp_name, self, create=False)

    @grace
    def create_experiment(self, xp_name, zip_file=None):
        assert (isinstance(xp_name, basestring))
        return TBExp(xp_name, self, zip_file=zip_file, create=True)

    @grace
    def remove_experiment(self, xp_name):
        assert (isinstance(xp_name, basestring))
        query = "/data?xp={}".format(quote_plus(xp_name))
        r = requests.delete(self.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    @grace
    def remove_all_experiments(self):
        xp_list = self.get_experiment_names()
        for xp_name in xp_list:
            self.remove_experiment(xp_name)


class TBExp(object):
    @grace
    def __init__(self, xp_name, client, zip_file=None, create=False):
        self.client = client
        self.xp_name = xp_name
        self.scalar_steps = collections.defaultdict(int)
        self.hist_steps = collections.defaultdict(int)

        if zip_file:
            if not create:
                msg = "Can only create a new experiment when "
                msg += "a zip_file is provided"
                raise ValueError(msg)
            self.__init_from_file(zip_file, True)

        elif create:
            self.__init_empty()

        else:
            self.__init_from_existing()

    # Initialisations
    @grace
    def __init_empty(self):
        query = "/data"
        r = requests.post(self.client.url + query, json=self.xp_name)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    @grace
    def __init_from_existing(self):
        query = "/data?xp={}".format(quote_plus(self.xp_name))
        r = requests.get(self.client.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

        # Retrieve the current step for existing metrics
        content = json.loads(r.text)
        self.__update_steps(content["scalars"], self.scalar_steps,
                            self.get_scalar_values)
        self.__update_steps(content["histograms"], self.hist_steps,
                            self.get_histogram_values)

    @grace
    def __init_from_file(self, zip_file, force=False):
        query = "/backup?xp={}&force={}".format(
            quote_plus(self.xp_name), force)
        fileobj = open(zip_file, 'rb')
        r = requests.post(
            self.client.url + query,
            data={"mysubmit": "Go"},
            files={"archive": ("backup.zip", fileobj)})
        fileobj.close()

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    # Scalar methods
    @grace
    def get_scalar_names(self):
        return self.__get_name_list("scalars")

    @grace
    def add_scalar_value(self, name, value, wall_time=-1, step=-1):
        if wall_time == -1:
            wall_time = time.time()
        if step == -1:
            step = self.scalar_steps[name]
            self.scalar_steps[name] += 1
        else:
            self.scalar_steps[name] = step + 1
        query = "/data/scalars?xp={}&name={}".format(
            quote_plus(self.xp_name), quote_plus(name))
        data = [wall_time, step, value]
        r = requests.post(self.client.url + query, json=data)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    @grace
    def add_text(self, name, text, wall_time=-1, step=-1):
        if wall_time == -1:
            wall_time = time.time()
        if step == -1:
            step = self.scalar_steps[name]
            self.scalar_steps[name] += 1
        else:
            self.scalar_steps[name] = step + 1
        query = "/data/texts?xp={}&name={}".format(
            quote_plus(self.xp_name), quote_plus(name))
        data = [wall_time, step, text]
        r = requests.post(self.client.url + query, json=data)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    @grace
    def add_image(self, name, image_path, wall_time=-1, step=-1):
        with open(image_path, 'rb') as image_file:
            image = base64.b64encode(image_file.read())
        if wall_time == -1:
            wall_time = time.time()
        if step == -1:
            step = self.scalar_steps[name]
            self.scalar_steps[name] += 1
        else:
            self.scalar_steps[name] = step + 1
        query = "/data/images?xp={}&name={}".format(
            quote_plus(self.xp_name), quote_plus(name))
        data = [wall_time, step, image]
        r = requests.post(self.client.url + query, json=data)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

    @grace
    def add_scalar_dict(self, data, wall_time=-1, step=-1):
        for name, value in data.items():
            if not isinstance(name, basestring):
                msg = "Scalar name should be a string, got: {}.".format(name)
                raise ValueError(msg)
            self.add_scalar_value(name, value, wall_time, step)

    @grace
    def get_scalar_values(self, name):
        query = "/data/scalars?xp={}&name={}".format(
            quote_plus(self.xp_name), quote_plus(name))

        r = requests.get(self.client.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

        return json.loads(r.text)

    # Histogram methods
    @grace
    def get_histogram_names(self):
        return self.__get_name_list("histograms")

    @grace
    def add_histogram_value(self,
                            name,
                            hist,
                            tobuild=False,
                            wall_time=-1,
                            step=-1):
        if wall_time == -1:
            wall_time = time.time()
        if step == -1:
            step = self.scalar_steps[name]
            self.scalar_steps[name] += 1
        else:
            self.scalar_steps[name] = step

        if not tobuild and (not isinstance(hist, dict)
                            or not self.__check_histogram_data(hist, tobuild)):
            raise ValueError("Data was not provided in a valid format!")

        if tobuild and (not isinstance(hist, list)):
            raise ValueError("Data was not provided in a valid format!")

        query = "/data/histograms?xp={}&name={}&tobuild={}".format(
            quote_plus(self.xp_name), quote_plus(name), tobuild)

        data = [wall_time, step, hist]
        r = requests.post(self.client.url + query, json=data)
        if not r.ok:
            raise ValueError("Something went wrong. Server sent: {}.".format(
                r.text))

    @grace
    def get_histogram_values(self, name):
        query = "/data/histograms?xp={}&name={}".format(
            quote_plus(self.xp_name), quote_plus(name))
        r = requests.get(self.client.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

        return json.loads(r.text)

    @grace
    def __check_histogram_data(self, data, tobuild):
        # TODO should use a schema here
        # Note: all of these are sorted already

        expected = ["bucket", "bucket_limit", "max", "min", "num"]
        expected2 = ["bucket", "bucket_limit", "max", "min", "num", "sum"]
        expected3 = [
            "bucket", "bucket_limit", "max", "min", "num", "sum", "sum_squares"
        ]
        expected4 = [
            "bucket", "bucket_limit", "max", "min", "num", "sum_squares"
        ]
        ks = tuple(data.keys())
        ks = sorted(ks)
        return (ks == expected or ks == expected2 or ks == expected3
                or ks == expected4)

    # Backup methods
    @grace
    def to_zip(self, filename=None):
        query = "/backup?xp={}".format(quote_plus(self.xp_name))
        r = requests.get(self.client.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

        if not filename:
            filename = "backup_" + self.xp_name + "_" + str(time.time())
        out = open(filename + ".zip", "wb")
        out.write(r.content)
        out.close()
        return filename + ".zip"

    # Helper methods
    @grace
    def __get_name_list(self, element_type):
        query = "/data?xp={}".format(quote_plus(self.xp_name))
        r = requests.get(self.client.url + query)

        if not r.ok:
            msg = "Something went wrong. Server sent: {}."
            raise ValueError(msg.format(r.text))

        return json.loads(r.text)[element_type]

    @grace
    def __update_steps(self, elements, steps_table, eval_function):
        for element in elements:
            values = eval_function(element)
            if len(values) > 0:
                steps_table[element] = values[-1][1] + 1
