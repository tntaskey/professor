#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
"""

from __future__ import with_statement

__author__ = 'Marcel Hellkamp'
__version__ = '0.12.9'
__license__ = 'MIT'

# The gevent server adapter needs to patch some modules before they are imported
# This is why we parse the commandline parameters here but handle them later
if __name__ == '__main__':
    from optparse import OptionParser
    _cmd_parser = OptionParser(usage="usage: %prog [options] package.module:app")
    _opt = _cmd_parser.add_option
    _opt("--version", action="store_true", help="show version number.")
    _opt("-b", "--bind", metavar="ADDRESS", help="bind socket to ADDRESS.")
    _opt("-s", "--server", default='wsgiref', help="use SERVER as backend.")
    _opt("-p", "--plugin", action="append", help="install additional plugin/s.")
    _opt("--debug", action="store_true", help="start server in debug mode.")
    _opt("--reload", action="store_true", help="auto-reload on file changes.")
    _cmd_options, _cmd_args = _cmd_parser.parse_args()
    if _cmd_options.server and _cmd_options.server.startswith('gevent'):
        import gevent.monkey; gevent.monkey.patch_all()

import base64, cgi, email.utils, functools, hmac, imp, itertools, mimetypes,\
        os, re, subprocess, sys, tempfile, threading, time, warnings

from datetime import date as datedate, datetime, timedelta
from tempfile import TemporaryFile
from traceback import format_exc, print_exc
from inspect import getargspec
from unicodedata import normalize


try: from simplejson import dumps as json_dumps, loads as json_lds
except ImportError: # pragma: no cover
    try: from json import dumps as json_dumps, loads as json_lds
    except ImportError:
        try: from django.utils.simplejson import dumps as json_dumps, loads as json_lds
        except ImportError:
            def json_dumps(data):
                raise ImportError("JSON support requires Python 2.6 or simplejson.")
            json_lds = json_dumps



# We now try to fix 2.5/2.6/3.1/3.2 incompatibilities.
# It ain't pretty but it works... Sorry for the mess.

py   = sys.version_info
py3k = py >= (3, 0, 0)
py25 = py <  (2, 6, 0)
py31 = (3, 1, 0) <= py < (3, 2, 0)

# Workaround for the missing "as" keyword in py3k.
def _e(): return sys.exc_info()[1]

# Workaround for the "print is a keyword/function" Python 2/3 dilemma
# and a fallback for mod_wsgi (resticts stdout/err attribute access)
try:
    _stdout, _stderr = sys.stdout.write, sys.stderr.write
except IOError:
    _stdout = lambda x: sys.stdout.write(x)
    _stderr = lambda x: sys.stderr.write(x)

# Lots of stdlib and builtin differences.
if py3k:
    import http.client as httplib
    import _thread as thread
    from urllib.parse import urljoin, SplitResult as UrlSplitResult
    from urllib.parse import urlencode, quote as urlquote, unquote as urlunquote
    urlunquote = functools.partial(urlunquote, encoding='latin1')
    from http.cookies import SimpleCookie
    from collections import MutableMapping as DictMixin
    import pickle
    from io import BytesIO
    from configparser import ConfigParser
    basestring = str
    unicode = str
    json_loads = lambda s: json_lds(touni(s))
    callable = lambda x: hasattr(x, '__call__')
    imap = map
    def _raise(*a): raise a[0](a[1]).with_traceback(a[2])
else: # 2.x
    import httplib
    import thread
    from urlparse import urljoin, SplitResult as UrlSplitResult
    from urllib import urlencode, quote as urlquote, unquote as urlunquote
    from Cookie import SimpleCookie
    from itertools import imap
    import cPickle as pickle
    from StringIO import StringIO as BytesIO
    from ConfigParser import SafeConfigParser as ConfigParser
    if py25:
        msg  = "Python 2.5 support may be dropped in future versions of Bottle."
        warnings.warn(msg, DeprecationWarning)
        from UserDict import DictMixin
        def next(it): return it.next()
        bytes = str
    else: # 2.6, 2.7
        from collections import MutableMapping as DictMixin
    unicode = unicode
    json_loads = json_lds
    eval(compile('def _raise(*a): raise a[0], a[1], a[2]', '<py3fix>', 'exec'))

# Some helpers for string/byte handling
def tob(s, enc='utf8'):
    return s.encode(enc) if isinstance(s, unicode) else bytes(s)
def touni(s, enc='utf8', err='strict'):
    return s.decode(enc, err) if isinstance(s, bytes) else unicode(s)
tonat = touni if py3k else tob

# 3.2 fixes cgi.FieldStorage to accept bytes (which makes a lot of sense).
# 3.1 needs a workaround.
if py31:
    from io import TextIOWrapper
    class NCTextIOWrapper(TextIOWrapper):
        def close(self): pass # Keep wrapped buffer open.


# A bug in functools causes it to break if the wrapper is an instance method
def update_wrapper(wrapper, wrapped, *a, **ka):
    try: functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError: pass



# These helpers are used at module level and need to be defined first.
# And yes, I know PEP-8, but sometimes a lower-case classname makes more sense.

def depr(message, hard=False):
    warnings.warn(message, DeprecationWarning, stacklevel=3)

def makelist(data): # This is just to handy
    if isinstance(data, (tuple, list, set, dict)): return list(data)
    elif data: return [data]
    else: return []


class DictProperty(object):
    ''' Property that maps to a key in a local dict-like attribute. '''
    def __init__(self, attr, key=None, read_only=False):
        self.attr, self.key, self.read_only = attr, key, read_only

    def __call__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter, self.key = func, self.key or func.__name__
        return self

    def __get__(self, obj, cls):
        if obj is None: return self
        key, storage = self.key, getattr(obj, self.attr)
        if key not in storage: storage[key] = self.getter(obj)
        return storage[key]

    def __set__(self, obj, value):
        if self.read_only: raise AttributeError("Read-Only property.")
        getattr(obj, self.attr)[self.key] = value

    def __delete__(self, obj):
        if self.read_only: raise AttributeError("Read-Only property.")
        del getattr(obj, self.attr)[self.key]


class cached_property(object):
    ''' A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. '''

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class lazy_attribute(object):
    ''' A property that caches itself to the class object. '''
    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value






###############################################################################
# Exceptions and Events ########################################################
###############################################################################


class BottleException(Exception):
    """ A base class for exceptions used by bottle. """
    pass






###############################################################################
# Routing ######################################################################
###############################################################################


class RouteError(BottleException):
    """ This is a base class for all routing related exceptions """


class RouteReset(BottleException):
    """ If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. """

class RouterUnknownModeError(RouteError): pass


class RouteSyntaxError(RouteError):
    """ The route parser found something not supported by this router. """


class RouteBuildError(RouteError):
    """ The route could not be built. """


def _re_flatten(p):
    ''' Turn all capturing groups in a regular expression pattern into
        non-capturing groups. '''
    if '(' not in p: return p
    return re.sub(r'(\\*)(\(\?P<[^>]+>|\((?!\?))',
        lambda m: m.group(0) if len(m.group(1)) % 2 else m.group(1) + '(?:', p)


class Router(object):
    ''' A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    '''

    default_pattern = '[^/]+'
    default_filter  = 're'

    #: The current CPython regexp implementation does not allow more
    #: than 99 matching groups per regular expression.
    _MAX_GROUPS_PER_PATTERN = 99

    def __init__(self, strict=False):
        self.rules    = [] # All rules in order
        self._groups  = {} # index of regexes to find them in dyna_routes
        self.builder  = {} # Data structure for the url builder
        self.static   = {} # Search structure for static routes
        self.dyna_routes   = {}
        self.dyna_regexes  = {} # Search structure for dynamic routes
        #: If true, static routes are no longer checked first.
        self.strict_order = strict
        self.filters = {
            're':    lambda conf:
                (_re_flatten(conf or self.default_pattern), None, None),
            'int':   lambda conf: (r'-?\d+', int, lambda x: str(int(x))),
            'float': lambda conf: (r'-?[\d.]+', float, lambda x: str(float(x))),
            'path':  lambda conf: (r'.+?', None, None)}

    def add_filter(self, name, func):
        ''' Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. '''
        self.filters[name] = func

    rule_syntax = re.compile('(\\\\*)'\
        '(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)'\
          '|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)'\
            '(?::((?:\\\\.|[^\\\\>]+)+)?)?)?>))')

    def _itertokens(self, rule):
        offset, prefix = 0, ''
        for match in self.rule_syntax.finditer(rule):
            prefix += rule[offset:match.start()]
            g = match.groups()
            if len(g[0])%2: # Escaped wildcard
                prefix += match.group(0)[len(g[0]):]
                offset = match.end()
                continue
            if prefix:
                yield prefix, None, None
            name, filtr, conf = g[4:7] if g[2] is None else g[1:4]
            yield name, filtr or 'default', conf or None
            offset, prefix = match.end(), ''
        if offset <= len(rule) or prefix:
            yield prefix+rule[offset:], None, None

    def add(self, rule, method, target, name=None):
        ''' Add a new rule or replace the target for an existing rule. '''
        anons     = 0    # Number of anonymous wildcards found
        keys      = []   # Names of keys
        pattern   = ''   # Regular expression pattern with named groups
        filters   = []   # Lists of wildcard input filters
        builder   = []   # Data structure for the URL builder
        is_static = True

        for key, mode, conf in self._itertokens(rule):
            if mode:
                is_static = False
                if mode == 'default': mode = self.default_filter
                mask, in_filter, out_filter = self.filters[mode](conf)
                if not key:
                    pattern += '(?:%s)' % mask
                    key = 'anon%d' % anons
                    anons += 1
                else:
                    pattern += '(?P<%s>%s)' % (key, mask)
                    keys.append(key)
                if in_filter: filters.append((key, in_filter))
                builder.append((key, out_filter or str))
            elif key:
                pattern += re.escape(key)
                builder.append((None, key))

        self.builder[rule] = builder
        if name: self.builder[name] = builder

        if is_static and not self.strict_order:
            self.static.setdefault(method, {})
            self.static[method][self.build(rule)] = (target, None)
            return

        try:
            re_pattern = re.compile('^(%s)$' % pattern)
            re_match = re_pattern.match
        except re.error:
            raise RouteSyntaxError("Could not add Route: %s (%s)" % (rule, _e()))

        if filters:
            def getargs(path):
                url_args = re_match(path).groupdict()
                for name, wildcard_filter in filters:
                    try:
                        url_args[name] = wildcard_filter(url_args[name])
                    except ValueError:
                        raise HTTPError(400, 'Path has wrong format.')
                return url_args
        elif re_pattern.groupindex:
            def getargs(path):
                return re_match(path).groupdict()
        else:
            getargs = None

        flatpat = _re_flatten(pattern)
        whole_rule = (rule, flatpat, target, getargs)

        if (flatpat, method) in self._groups:
            if DEBUG:
                msg = 'Route <%s %s> overwrites a previously defined route'
                warnings.warn(msg % (method, rule), RuntimeWarning)
            self.dyna_routes[method][self._groups[flatpat, method]] = whole_rule
        else:
            self.dyna_routes.setdefault(method, []).append(whole_rule)
            self._groups[flatpat, method] = len(self.dyna_routes[method]) - 1

        self._compile(method)

    def _compile(self, method):
        all_rules = self.dyna_routes[method]
        comborules = self.dyna_regexes[method] = []
        maxgroups = self._MAX_GROUPS_PER_PATTERN
        for x in range(0, len(all_rules), maxgroups):
            some = all_rules[x:x+maxgroups]
            combined = (flatpat for (_, flatpat, _, _) in some)
            combined = '|'.join('(^%s$)' % flatpat for flatpat in combined)
            combined = re.compile(combined).match
            rules = [(target, getargs) for (_, _, target, getargs) in some]
            comborules.append((combined, rules))

    def build(self, _name, *anons, **query):
        ''' Build an URL by filling the wildcards in a rule. '''
        builder = self.builder.get(_name)
        if not builder: raise RouteBuildError("No route with that name.", _name)
        try:
            for i, value in enumerate(anons): query['anon%d'%i] = value
            url = ''.join([f(query.pop(n)) if n else f for (n,f) in builder])
            return url if not query else url+'?'+urlencode(query)
        except KeyError:
            raise RouteBuildError('Missing URL argument: %r' % _e().args[0])

    def match(self, environ):
        ''' Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). '''
        verb = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'] or '/'
        target = None
        if verb == 'HEAD':
            methods = ['PROXY', verb, 'GET', 'ANY']
        else:
            methods = ['PROXY', verb, 'ANY']

        for method in methods:
            if method in self.static and path in self.static[method]:
                target, getargs = self.static[method][path]
                return target, getargs(path) if getargs else {}
            elif method in self.dyna_regexes:
                for combined, rules in self.dyna_regexes[method]:
                    match = combined(path)
                    if match:
                        target, getargs = rules[match.lastindex - 1]
                        return target, getargs(path) if getargs else {}

        # No matching route found. Collect alternative methods for 405 response
        allowed = set([])
        nocheck = set(methods)
        for method in set(self.static) - nocheck:
            if path in self.static[method]:
                allowed.add(verb)
        for method in set(self.dyna_regexes) - allowed - nocheck:
            for combined, rules in self.dyna_regexes[method]:
                match = combined(path)
                if match:
                    allowed.add(method)
        if allowed:
            allow_header = ",".join(sorted(allowed))
            raise HTTPError(405, "Method not allowed.", Allow=allow_header)

        # No matching route and no alternative method found. We give up
        raise HTTPError(404, "Not found: " + repr(path))






class Route(object):
    ''' This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    '''

    def __init__(self, app, rule, method, callback, name=None,
                 plugins=None, skiplist=None, **config):
        #: The application this route is installed to.
        self.app = app
        #: The path-rule string (e.g. ``/wiki/:page``).
        self.rule = rule
        #: The HTTP method as a string (e.g. ``GET``).
        self.method = method
        #: The original callback with no plugins applied. Useful for introspection.
        self.callback = callback
        #: The name of the route (if specified) or ``None``.
        self.name = name or None
        #: A list of route-specific plugins (see :meth:`Bottle.route`).
        self.plugins = plugins or []
        #: A list of plugins to not apply to this route (see :meth:`Bottle.route`).
        self.skiplist = skiplist or []
        #: Additional keyword arguments passed to the :meth:`Bottle.route`
        #: decorator are stored in this dictionary. Used for route-specific
        #: plugin configuration and meta-data.
        self.config = ConfigDict().load_dict(config, make_namespaces=True)

    def __call__(self, *a, **ka):
        depr("Some APIs changed to return Route() instances instead of"\
             " callables. Make sure to use the Route.call method and not to"\
             " call Route instances directly.") #0.12
        return self.call(*a, **ka)

    @cached_property
    def call(self):
        ''' The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.'''
        return self._make_callback()

    def reset(self):
        ''' Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. '''
        self.__dict__.pop('call', None)

    def prepare(self):
        ''' Do all on-demand work immediately (useful for debugging).'''
        self.call

    @property
    def _context(self):
        depr('Switch to Plugin API v2 and access the Route object directly.')  #0.12
        return dict(rule=self.rule, method=self.method, callback=self.callback,
                    name=self.name, app=self.app, config=self.config,
                    apply=self.plugins, skip=self.skiplist)

    def all_plugins(self):
        ''' Yield all Plugins affecting this route. '''
        unique = set()
        for p in reversed(self.app.plugins + self.plugins):
            if True in self.skiplist: break
            name = getattr(p, 'name', False)
            if name and (name in self.skiplist or name in unique): continue
            if p in self.skiplist or type(p) in self.skiplist: continue
            if name: unique.add(name)
            yield p

    def _make_callback(self):
        callback = self.callback
        for plugin in self.all_plugins():
            try:
                if hasattr(plugin, 'apply'):
                    api = getattr(plugin, 'api', 1)
                    context = self if api > 1 else self._context
                    callback = plugin.apply(callback, context)
                else:
                    callback = plugin(callback)
            except RouteReset: # Try again with changed configuration.
                return self._make_callback()
            if not callback is self.callback:
                update_wrapper(callback, self.callback)
        return callback

    def get_undecorated_callback(self):
        ''' Return the callback. If the callback is a decorated function, try to
            recover the original function. '''
        func = self.callback
        func = getattr(func, '__func__' if py3k else 'im_func', func)
        closure_attr = '__closure__' if py3k else 'func_closure'
        while hasattr(func, closure_attr) and getattr(func, closure_attr):
            func = getattr(func, closure_attr)[0].cell_contents
        return func

    def get_callback_args(self):
        ''' Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. '''
        return getargspec(self.get_undecorated_callback())[0]

    def get_config(self, key, default=None):
        ''' Lookup a config field and return its value, first checking the
            route.config, then route.app.config.'''
        for conf in (self.config, self.app.conifg):
            if key in conf: return conf[key]
        return default

    def __repr__(self):
        cb = self.get_undecorated_callback()
        return '<%s %r %r>' % (self.method, self.rule, cb)






###############################################################################
# Application Object ###########################################################
###############################################################################


class Bottle(object):
    """ Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    """

    def __init__(self, catchall=True, autojson=True):

        #: A :class:`ConfigDict` for app specific configuration.
        self.config = ConfigDict()
        self.config._on_change = functools.partial(self.trigger_hook, 'config')
        self.config.meta_set('autojson', 'validate', bool)
        self.config.meta_set('catchall', 'validate', bool)
        self.config['catchall'] = catchall
        self.config['autojson'] = autojson

        #: A :class:`ResourceManager` for application files
        self.resources = ResourceManager()

        self.routes = [] # List of installed :class:`Route` instances.
        self.router = Router() # Maps requests to :class:`Route` instances.
        self.error_handler = {}

        # Core plugins
        self.plugins = [] # List of installed plugins.
        if self.config['autojson']:
            self.install(JSONPlugin())
        self.install(TemplatePlugin())

    #: If true, most exceptions are caught and returned as :exc:`HTTPError`
    catchall = DictProperty('config', 'catchall')

    __hook_names = 'before_request', 'after_request', 'app_reset', 'config'
    __hook_reversed = 'after_request'

    @cached_property
    def _hooks(self):
        return dict((name, []) for name in self.__hook_names)

    def add_hook(self, name, func):
        ''' Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        '''
        if name in self.__hook_reversed:
            self._hooks[name].insert(0, func)
        else:
            self._hooks[name].append(func)

    def remove_hook(self, name, func):
        ''' Remove a callback from a hook. '''
        if name in self._hooks and func in self._hooks[name]:
            self._hooks[name].remove(func)
            return True

    def trigger_hook(self, __name, *args, **kwargs):
        ''' Trigger a hook and return a list of results. '''
        return [hook(*args, **kwargs) for hook in self._hooks[__name][:]]

    def hook(self, name):
        """ Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details."""
        def decorator(func):
            self.add_hook(name, func)
            return func
        return decorator

    def mount(self, prefix, app, **options):
        ''' Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        '''
        if isinstance(app, basestring):
            depr('Parameter order of Bottle.mount() changed.', True) # 0.10

        segments = [p for p in prefix.split('/') if p]
        if not segments: raise ValueError('Empty path prefix.')
        path_depth = len(segments)

        def mountpoint_wrapper():
            try:
                request.path_shift(path_depth)
                rs = HTTPResponse([])
                def start_response(status, headerlist, exc_info=None):
                    if exc_info:
                        try:
                            _raise(*exc_info)
                        finally:
                            exc_info = None
                    rs.status = status
                    for name, value in headerlist: rs.add_header(name, value)
                    return rs.body.append
                body = app(request.environ, start_response)
                if body and rs.body: body = itertools.chain(rs.body, body)
                rs.body = body or rs.body
                return rs
            finally:
                request.path_shift(-path_depth)

        options.setdefault('skip', True)
        options.setdefault('method', 'PROXY')
        options.setdefault('mountpoint', {'prefix': prefix, 'target': app})
        options['callback'] = mountpoint_wrapper

        self.route('/%s/<:re:.*>' % '/'.join(segments), **options)
        if not prefix.endswith('/'):
            self.route('/' + '/'.join(segments), **options)

    def merge(self, routes):
        ''' Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. '''
        if isinstance(routes, Bottle):
            routes = routes.routes
        for route in routes:
            self.add_route(route)

    def install(self, plugin):
        ''' Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        '''
        if hasattr(plugin, 'setup'): plugin.setup(self)
        if not callable(plugin) and not hasattr(plugin, 'apply'):
            raise TypeError("Plugins must be callable or implement .apply()")
        self.plugins.append(plugin)
        self.reset()
        return plugin

    def uninstall(self, plugin):
        ''' Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. '''
        removed, remove = [], plugin
        for i, plugin in list(enumerate(self.plugins))[::-1]:
            if remove is True or remove is plugin or remove is type(plugin) \
            or getattr(plugin, 'name', True) == remove:
                removed.append(plugin)
                del self.plugins[i]
                if hasattr(plugin, 'close'): plugin.close()
        if removed: self.reset()
        return removed

    def reset(self, route=None):
        ''' Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. '''
        if route is None: routes = self.routes
        elif isinstance(route, Route): routes = [route]
        else: routes = [self.routes[route]]
        for route in routes: route.reset()
        if DEBUG:
            for route in routes: route.prepare()
        self.trigger_hook('app_reset')

    def close(self):
        ''' Close the application and all installed plugins. '''
        for plugin in self.plugins:
            if hasattr(plugin, 'close'): plugin.close()
        self.stopped = True

    def run(self, **kwargs):
        ''' Calls :func:`run` with the same parameters. '''
        run(self, **kwargs)

    def match(self, environ):
        """ Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match."""
        return self.router.match(environ)

    def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        scriptname = request.environ.get('SCRIPT_NAME', '').strip('/') + '/'
        location = self.router.build(routename, **kargs).lstrip('/')
        return urljoin(urljoin('/', scriptname), location)

    def add_route(self, route):
        ''' Add a route object, but do not change the :data:`Route.app`
            attribute.'''
        self.routes.append(route)
        self.router.add(route.rule, route.method, route, name=route.name)
        if DEBUG: route.prepare()

    def route(self, path=None, method='GET', callback=None, name=None,
              apply=None, skip=None, **config):
        """ A decorator to bind a function to a request URL. Example::

                @app.route('/hello/:name')
                def hello(name):
                    return 'Hello %s' % name

            The ``:name`` part is a wildcard. See :class:`Router` for syntax
            details.

            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.

            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        """
        if callable(path): path, callback = None, path
        plugins = makelist(apply)
        skiplist = makelist(skip)
        def decorator(callback):
            # TODO: Documentation and tests
            if isinstance(callback, basestring): callback = load(callback)
            for rule in makelist(path) or yieldroutes(callback):
                for verb in makelist(method):
                    verb = verb.upper()
                    route = Route(self, rule, verb, callback, name=name,
                                  plugins=plugins, skiplist=skiplist, **config)
                    self.add_route(route)
            return callback
        return decorator(callback) if callback else decorator

    def get(self, path=None, method='GET', **options):
        """ Equals :meth:`route`. """
        return self.route(path, method, **options)

    def post(self, path=None, method='POST', **options):
        """ Equals :meth:`route` with a ``POST`` method parameter. """
        return self.route(path, method, **options)

    def put(self, path=None, method='PUT', **options):
        """ Equals :meth:`route` with a ``PUT`` method parameter. """
        return self.route(path, method, **options)

    def delete(self, path=None, method='DELETE', **options):
        """ Equals :meth:`route` with a ``DELETE`` method parameter. """
        return self.route(path, method, **options)

    def error(self, code=500):
        """ Decorator: Register an output handler for a HTTP error code"""
        def wrapper(handler):
            self.error_handler[int(code)] = handler
            return handler
        return wrapper

    def default_error_handler(self, res):
        return tob(template(ERROR_PAGE_TEMPLATE, e=res))

    def _handle(self, environ):
        path = environ['bottle.raw_path'] = environ['PATH_INFO']
        if py3k:
            try:
                environ['PATH_INFO'] = path.encode('latin1').decode('utf8')
            except UnicodeError:
                return HTTPError(400, 'Invalid path string. Expected UTF-8')

        try:
            environ['bottle.app'] = self
            request.bind(environ)
            response.bind()
            try:
                self.trigger_hook('before_request')
                route, args = self.router.match(environ)
                environ['route.handle'] = route
                environ['bottle.route'] = route
                environ['route.url_args'] = args
                return route.call(**args)
            finally:
                self.trigger_hook('after_request')

        except HTTPResponse:
            return _e()
        except RouteReset:
            route.reset()
            return self._handle(environ)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            stacktrace = format_exc()
            environ['wsgi.errors'].write(stacktrace)
            return HTTPError(500, "Internal Server Error", _e(), stacktrace)

    def _cast(self, out, peek=None):
        """ Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        """

        # Empty output is done here
        if not out:
            if 'Content-Length' not in response:
                response['Content-Length'] = 0
            return []
        # Join lists of byte or unicode strings. Mixed lists are NOT supported
        if isinstance(out, (tuple, list))\
        and isinstance(out[0], (bytes, unicode)):
            out = out[0][0:0].join(out) # b'abc'[0:0] -> b''
        # Encode unicode strings
        if isinstance(out, unicode):
            out = out.encode(response.charset)
        # Byte Strings are just returned
        if isinstance(out, bytes):
            if 'Content-Length' not in response:
                response['Content-Length'] = len(out)
            return [out]
        # HTTPError or HTTPException (recursive, because they may wrap anything)
        # TODO: Handle these explicitly in handle() or make them iterable.
        if isinstance(out, HTTPError):
            out.apply(response)
            out = self.error_handler.get(out.status_code, self.default_error_handler)(out)
            return self._cast(out)
        if isinstance(out, HTTPResponse):
            out.apply(response)
            return self._cast(out.body)

        # File-like objects.
        if hasattr(out, 'read'):
            if 'wsgi.file_wrapper' in request.environ:
                return request.environ['wsgi.file_wrapper'](out)
            elif hasattr(out, 'close') or not hasattr(out, '__iter__'):
                return WSGIFileWrapper(out)

        # Handle Iterables. We peek into them to detect their inner type.
        try:
            iout = iter(out)
            first = next(iout)
            while not first:
                first = next(iout)
        except StopIteration:
            return self._cast('')
        except HTTPResponse:
            first = _e()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            first = HTTPError(500, 'Unhandled exception', _e(), format_exc())

        # These are the inner types allowed in iterator or generator objects.
        if isinstance(first, HTTPResponse):
            return self._cast(first)
        elif isinstance(first, bytes):
            new_iter = itertools.chain([first], iout)
        elif isinstance(first, unicode):
            encoder = lambda x: x.encode(response.charset)
            new_iter = imap(encoder, itertools.chain([first], iout))
        else:
            msg = 'Unsupported response type: %s' % type(first)
            return self._cast(HTTPError(500, msg))
        if hasattr(out, 'close'):
            new_iter = _closeiter(new_iter, out.close)
        return new_iter

    def wsgi(self, environ, start_response):
        """ The bottle WSGI-interface. """
        try:
            out = self._cast(self._handle(environ))
            # rfc2616 section 4.3
            if response._status_code in (100, 101, 204, 304)\
            or environ['REQUEST_METHOD'] == 'HEAD':
                if hasattr(out, 'close'): out.close()
                out = []
            start_response(response._status_line, response.headerlist)
            return out
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            err = '<h1>Critical error while processing request: %s</h1>' \
                  % html_escape(environ.get('PATH_INFO', '/'))
            if DEBUG:
                err += '<h2>Error:</h2>\n<pre>\n%s\n</pre>\n' \
                       '<h2>Traceback:</h2>\n<pre>\n%s\n</pre>\n' \
                       % (html_escape(repr(_e())), html_escape(format_exc()))
            environ['wsgi.errors'].write(err)
            headers = [('Content-Type', 'text/html; charset=UTF-8')]
            start_response('500 INTERNAL SERVER ERROR', headers, sys.exc_info())
            return [tob(err)]

    def __call__(self, environ, start_response):
        ''' Each instance of :class:'Bottle' is a WSGI application. '''
        return self.wsgi(environ, start_response)






###############################################################################
# HTTP and WSGI Tools ##########################################################
###############################################################################

class BaseRequest(object):
    """ A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    """

    __slots__ = ('environ')

    #: Maximum size of memory buffer for :attr:`body` in bytes.
    MEMFILE_MAX = 102400

    def __init__(self, environ=None):
        """ Wrap a WSGI environ dictionary. """
        #: The wrapped WSGI environ dictionary. This is the only real attribute.
        #: All other attributes actually are read-only properties.
        self.environ = {} if environ is None else environ
        self.environ['bottle.request'] = self

    @DictProperty('environ', 'bottle.app', read_only=True)
    def app(self):
        ''' Bottle application handling this request. '''
        raise RuntimeError('This request is not connected to an application.')

    @DictProperty('environ', 'bottle.route', read_only=True)
    def route(self):
        """ The bottle :class:`Route` object that matches this request. """
        raise RuntimeError('This request is not connected to a route.')

    @DictProperty('environ', 'route.url_args', read_only=True)
    def url_args(self):
        """ The arguments extracted from the URL. """
        raise RuntimeError('This request is not connected to a route.')

    @property
    def path(self):
        ''' The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). '''
        return '/' + self.environ.get('PATH_INFO','').lstrip('/')

    @property
    def method(self):
        ''' The ``REQUEST_METHOD`` value as an uppercase string. '''
        return self.environ.get('REQUEST_METHOD', 'GET').upper()

    @DictProperty('environ', 'bottle.request.headers', read_only=True)
    def headers(self):
        ''' A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. '''
        return WSGIHeaderDict(self.environ)

    def get_header(self, name, default=None):
        ''' Return the value of a request header, or a given default value. '''
        return self.headers.get(name, default)

    @DictProperty('environ', 'bottle.request.cookies', read_only=True)
    def cookies(self):
        """ Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. """
        cookies = SimpleCookie(self.environ.get('HTTP_COOKIE','')).values()
        return FormsDict((c.key, c.value) for c in cookies)

    def get_cookie(self, key, default=None, secret=None):
        """ Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. """
        value = self.cookies.get(key)
        if secret and value:
            dec = cookie_decode(value, secret) # (key, value) tuple or None
            return dec[1] if dec and dec[0] == key else default
        return value or default

    @DictProperty('environ', 'bottle.request.query', read_only=True)
    def query(self):
        ''' The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. '''
        get = self.environ['bottle.get'] = FormsDict()
        pairs = _parse_qsl(self.environ.get('QUERY_STRING', ''))
        for key, value in pairs:
            get[key] = value
        return get

    @DictProperty('environ', 'bottle.request.forms', read_only=True)
    def forms(self):
        """ Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. """
        forms = FormsDict()
        for name, item in self.POST.allitems():
            if not isinstance(item, FileUpload):
                forms[name] = item
        return forms

    @DictProperty('environ', 'bottle.request.params', read_only=True)
    def params(self):
        """ A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. """
        params = FormsDict()
        for key, value in self.query.allitems():
            params[key] = value
        for key, value in self.forms.allitems():
            params[key] = value
        return params

    @DictProperty('environ', 'bottle.request.files', read_only=True)
    def files(self):
        """ File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        """
        files = FormsDict()
        for name, item in self.POST.allitems():
            if isinstance(item, FileUpload):
                files[name] = item
        return files

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        ''' If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. '''
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype == 'application/json':
            b = self._get_body_string()
            if not b:
                return None
            return json_loads(b)
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    def _iter_chunked(self, read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        read_func = self.environ['wsgi.input'].read
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self):
        ''' read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. '''
        clen = self.content_length
        if clen > self.MEMFILE_MAX:
            raise HTTPError(413, 'Request to large')
        if clen < 0: clen = self.MEMFILE_MAX + 1
        data = self.body.read(clen)
        if len(data) > self.MEMFILE_MAX: # Fail fast
            raise HTTPError(413, 'Request to large')
        return data

    @property
    def body(self):
        """ The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. """
        self._body.seek(0)
        return self._body

    @property
    def chunked(self):
        ''' True if Chunked transfer encoding was. '''
        return 'chunked' in self.environ.get('HTTP_TRANSFER_ENCODING', '').lower()

    #: An alias for :attr:`query`.
    GET = query

    @DictProperty('environ', 'bottle.request.post', read_only=True)
    def POST(self):
        """ The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        """
        post = FormsDict()
        # We default to application/x-www-form-urlencoded for everything that
        # is not multipart and take the fast path (also: 3.1 workaround)
        if not self.content_type.startswith('multipart/'):
            pairs = _parse_qsl(tonat(self._get_body_string(), 'latin1'))
            for key, value in pairs:
                post[key] = value
            return post

        safe_env = {'QUERY_STRING':''} # Build a safe environment for cgi
        for key in ('REQUEST_METHOD', 'CONTENT_TYPE', 'CONTENT_LENGTH'):
            if key in self.environ: safe_env[key] = self.environ[key]
        args = dict(fp=self.body, environ=safe_env, keep_blank_values=True)
        if py31:
            args['fp'] = NCTextIOWrapper(args['fp'], encoding='utf8',
                                         newline='\n')
        elif py3k:
            args['encoding'] = 'utf8'
        data = cgi.FieldStorage(**args)
        self['_cgi.FieldStorage'] = data #http://bugs.python.org/issue18394#msg207958
        data = data.list or []
        for item in data:
            if item.filename:
                post[item.name] = FileUpload(item.file, item.name,
                                             item.filename, item.headers)
            else:
                post[item.name] = item.value
        return post

    @property
    def url(self):
        """ The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. """
        return self.urlparts.geturl()

    @DictProperty('environ', 'bottle.request.urlparts', read_only=True)
    def urlparts(self):
        ''' The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. '''
        env = self.environ
        http = env.get('HTTP_X_FORWARDED_PROTO') or env.get('wsgi.url_scheme', 'http')
        host = env.get('HTTP_X_FORWARDED_HOST') or env.get('HTTP_HOST')
        if not host:
            # HTTP 1.1 requires a Host-header. This is for HTTP/1.0 clients.
            host = env.get('SERVER_NAME', '127.0.0.1')
            port = env.get('SERVER_PORT')
            if port and port != ('80' if http == 'http' else '443'):
                host += ':' + port
        path = urlquote(self.fullpath)
        return UrlSplitResult(http, host, path, env.get('QUERY_STRING'), '')

    @property
    def fullpath(self):
        """ Request path including :attr:`script_name` (if present). """
        return urljoin(self.script_name, self.path.lstrip('/'))

    @property
    def query_string(self):
        """ The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. """
        return self.environ.get('QUERY_STRING', '')

    @property
    def script_name(self):
        ''' The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. '''
        script_name = self.environ.get('SCRIPT_NAME', '').strip('/')
        return '/' + script_name + '/' if script_name else '/'

    def path_shift(self, shift=1):
        ''' Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        '''
        script = self.environ.get('SCRIPT_NAME','/')
        self['SCRIPT_NAME'], self['PATH_INFO'] = path_shift(script, self.path, shift)

    @property
    def content_length(self):
        ''' The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. '''
        return int(self.environ.get('CONTENT_LENGTH') or -1)

    @property
    def content_type(self):
        ''' The Content-Type header as a lowercase-string (default: empty). '''
        return self.environ.get('CONTENT_TYPE', '').lower()

    @property
    def is_xhr(self):
        ''' True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). '''
        requested_with = self.environ.get('HTTP_X_REQUESTED_WITH','')
        return requested_with.lower() == 'xmlhttprequest'

    @property
    def is_ajax(self):
        ''' Alias for :attr:`is_xhr`. "Ajax" is not the right term. '''
        return self.is_xhr

    @property
    def auth(self):
        """ HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. """
        basic = parse_auth(self.environ.get('HTTP_AUTHORIZATION',''))
        if basic: return basic
        ruser = self.environ.get('REMOTE_USER')
        if ruser: return (ruser, None)
        return None

    @property
    def remote_route(self):
        """ A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. """
        proxy = self.environ.get('HTTP_X_FORWARDED_FOR')
        if proxy: return [ip.strip() for ip in proxy.split(',')]
        remote = self.environ.get('REMOTE_ADDR')
        return [remote] if remote else []

    @property
    def remote_addr(self):
        """ The client IP as a string. Note that this information can be forged
            by malicious clients. """
        route = self.remote_route
        return route[0] if route else None

    def copy(self):
        """ Return a new :class:`Request` with a shallow :attr:`environ` copy. """
        return Request(self.environ.copy())

    def get(self, value, default=None): return self.environ.get(value, default)
    def __getitem__(self, key): return self.environ[key]
    def __delitem__(self, key): self[key] = ""; del(self.environ[key])
    def __iter__(self): return iter(self.environ)
    def __len__(self): return len(self.environ)
    def keys(self): return self.environ.keys()
    def __setitem__(self, key, value):
        """ Change an environ value and clear all caches that depend on it. """

        if self.environ.get('bottle.request.readonly'):
            raise KeyError('The environ dictionary is read-only.')

        self.environ[key] = value
        todelete = ()

        if key == 'wsgi.input':
            todelete = ('body', 'forms', 'files', 'params', 'post', 'json')
        elif key == 'QUERY_STRING':
            todelete = ('query', 'params')
        elif key.startswith('HTTP_'):
            todelete = ('headers', 'cookies')

        for key in todelete:
            self.environ.pop('bottle.request.'+key, None)

    def __repr__(self):
        return '<%s: %s %s>' % (self.__class__.__name__, self.method, self.url)

    def __getattr__(self, name):
        ''' Search in self.environ for additional user defined attributes. '''
        try:
            var = self.environ['bottle.request.ext.%s'%name]
            return var.__get__(self) if hasattr(var, '__get__') else var
        except KeyError:
            raise AttributeError('Attribute %r not defined.' % name)

    def __setattr__(self, name, value):
        if name == 'environ': return object.__setattr__(self, name, value)
        self.environ['bottle.request.ext.%s'%name] = value




def _hkey(s):
    return s.title().replace('_','-')


class HeaderProperty(object):
    def __init__(self, name, reader=None, writer=str, default=''):
        self.name, self.default = name, default
        self.reader, self.writer = reader, writer
        self.__doc__ = 'Current value of the %r header.' % name.title()

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.headers.get(self.name, self.default)
        return self.reader(value) if self.reader else value

    def __set__(self, obj, value):
        obj.headers[self.name] = self.writer(value)

    def __delete__(self, obj):
        del obj.headers[self.name]


class BaseResponse(object):
    """ Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    """

    default_status = 200
    default_content_type = 'text/html; charset=UTF-8'

    # Header blacklist for specific response codes
    # (rfc2616 section 10.2.3 and 10.3.5)
    bad_headers = {
        204: set(('Content-Type',)),
        304: set(('Allow', 'Content-Encoding', 'Content-Language',
                  'Content-Length', 'Content-Range', 'Content-Type',
                  'Content-Md5', 'Last-Modified'))}

    def __init__(self, body='', status=None, headers=None, **more_headers):
        self._cookies = None
        self._headers = {}
        self.body = body
        self.status = status or self.default_status
        if headers:
            if isinstance(headers, dict):
                headers = headers.items()
            for name, value in headers:
                self.add_header(name, value)
        if more_headers:
            for name, value in more_headers.items():
                self.add_header(name, value)

    def copy(self, cls=None):
        ''' Returns a copy of self. '''
        cls = cls or BaseResponse
        assert issubclass(cls, BaseResponse)
        copy = cls()
        copy.status = self.status
        copy._headers = dict((k, v[:]) for (k, v) in self._headers.items())
        if self._cookies:
            copy._cookies = SimpleCookie()
            copy._cookies.load(self._cookies.output(header=''))
        return copy

    def __iter__(self):
        return iter(self.body)

    def close(self):
        if hasattr(self.body, 'close'):
            self.body.close()

    @property
    def status_line(self):
        ''' The HTTP status line as a string (e.g. ``404 Not Found``).'''
        return self._status_line

    @property
    def status_code(self):
        ''' The HTTP status code as an integer (e.g. 404).'''
        return self._status_code

    def _set_status(self, status):
        if isinstance(status, int):
            code, status = status, _HTTP_STATUS_LINES.get(status)
        elif ' ' in status:
            status = status.strip()
            code   = int(status.split()[0])
        else:
            raise ValueError('String status line without a reason phrase.')
        if not 100 <= code <= 999: raise ValueError('Status code out of range.')
        self._status_code = code
        self._status_line = str(status or ('%d Unknown' % code))

    def _get_status(self):
        return self._status_line

    status = property(_get_status, _set_status, None,
        ''' A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. ''')
    del _get_status, _set_status

    @property
    def headers(self):
        ''' An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. '''
        hdict = HeaderDict()
        hdict.dict = self._headers
        return hdict

    def __contains__(self, name): return _hkey(name) in self._headers
    def __delitem__(self, name):  del self._headers[_hkey(name)]
    def __getitem__(self, name):  return self._headers[_hkey(name)][-1]
    def __setitem__(self, name, value): self._headers[_hkey(name)] = [str(value)]

    def get_header(self, name, default=None):
        ''' Return the value of a previously defined header. If there is no
            header with that name, return a default value. '''
        return self._headers.get(_hkey(name), [default])[-1]

    def set_header(self, name, value):
        ''' Create a new response header, replacing any previously defined
            headers with the same name. '''
        self._headers[_hkey(name)] = [str(value)]

    def add_header(self, name, value):
        ''' Add an additional response header, not removing duplicates. '''
        self._headers.setdefault(_hkey(name), []).append(str(value))

    def iter_headers(self):
        ''' Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. '''
        return self.headerlist

    @property
    def headerlist(self):
        ''' WSGI conform list of (header, value) tuples. '''
        out = []
        headers = list(self._headers.items())
        if 'Content-Type' not in self._headers:
            headers.append(('Content-Type', [self.default_content_type]))
        if self._status_code in self.bad_headers:
            bad_headers = self.bad_headers[self._status_code]
            headers = [h for h in headers if h[0] not in bad_headers]
        out += [(name, val) for name, vals in headers for val in vals]
        if self._cookies:
            for c in self._cookies.values():
                out.append(('Set-Cookie', c.OutputString()))
        return out

    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int)
    expires = HeaderProperty('Expires',
        reader=lambda x: datetime.utcfromtimestamp(parse_date(x)),
        writer=lambda x: http_date(x))

    @property
    def charset(self, default='UTF-8'):
        """ Return the charset specified in the content-type header (default: utf8). """
        if 'charset=' in self.content_type:
            return self.content_type.split('charset=')[-1].split(';')[0].strip()
        return default

    def set_cookie(self, name, value, secret=None, **options):
        ''' Create a new cookie or replace an old one. If the `secret` parameter is
            set, create a `Signed Cookie` (described below).

            :param name: the name of the cookie.
            :param value: the value of the cookie.
            :param secret: a signature key required for signed cookies.

            Additionally, this method accepts all RFC 2109 attributes that are
            supported by :class:`cookie.Morsel`, including:

            :param max_age: maximum age in seconds. (default: None)
            :param expires: a datetime object or UNIX timestamp. (default: None)
            :param domain: the domain that is allowed to read the cookie.
              (default: current domain)
            :param path: limits the cookie to a given path (default: current path)
            :param secure: limit the cookie to HTTPS connections (default: off).
            :param httponly: prevents client-side javascript to read this cookie
              (default: off, requires Python 2.6 or newer).

            If neither `expires` nor `max_age` is set (default), the cookie will
            expire at the end of the browser session (as soon as the browser
            window is closed).

            Signed cookies may store any pickle-able object and are
            cryptographically signed to prevent manipulation. Keep in mind that
            cookies are limited to 4kb in most browsers.

            Warning: Signed cookies are not encrypted (the client can still see
            the content) and not copy-protected (the client can restore an old
            cookie). The main intention is to make pickling and unpickling
            save, not to store secret information at client side.
        '''
        if not self._cookies:
            self._cookies = SimpleCookie()

        if secret:
            value = touni(cookie_encode((name, value), secret))
        elif not isinstance(value, basestring):
            raise TypeError('Secret key missing for non-string Cookie.')

        if len(value) > 4096: raise ValueError('Cookie value to long.')
        self._cookies[name] = value

        for key, value in options.items():
            if key == 'max_age':
                if isinstance(value, timedelta):
                    value = value.seconds + value.days * 24 * 3600
            if key == 'expires':
                if isinstance(value, (datedate, datetime)):
                    value = value.timetuple()
                elif isinstance(value, (int, float)):
                    value = time.gmtime(value)
                value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
            self._cookies[name][key.replace('_', '-')] = value

    def delete_cookie(self, key, **kwargs):
        ''' Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. '''
        kwargs['max_age'] = -1
        kwargs['expires'] = 0
        self.set_cookie(key, '', **kwargs)

    def __repr__(self):
        out = ''
        for name, value in self.headerlist:
            out += '%s: %s\n' % (name.title(), value.strip())
        return out


def local_property(name=None):
    if name: depr('local_property() is deprecated and will be removed.') #0.12
    ls = threading.local()
    def fget(self):
        try: return ls.var
        except AttributeError:
            raise RuntimeError("Request context not initialized.")
    def fset(self, value): ls.var = value
    def fdel(self): del ls.var
    return property(fget, fset, fdel, 'Thread-local property')


class LocalRequest(BaseRequest):
    ''' A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). '''
    bind = BaseRequest.__init__
    environ = local_property()


class LocalResponse(BaseResponse):
    ''' A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    '''
    bind = BaseResponse.__init__
    _status_line = local_property()
    _status_code = local_property()
    _cookies     = local_property()
    _headers     = local_property()
    body         = local_property()


Request = BaseRequest
Response = BaseResponse


class HTTPResponse(Response, BottleException):
    def __init__(self, body='', status=None, headers=None, **more_headers):
        super(HTTPResponse, self).__init__(body, status, headers, **more_headers)

    def apply(self, response):
        response._status_code = self._status_code
        response._status_line = self._status_line
        response._headers = self._headers
        response._cookies = self._cookies
        response.body = self.body


class HTTPError(HTTPResponse):
    default_status = 500
    def __init__(self, status=None, body=None, exception=None, traceback=None,
                 **options):
        self.exception = exception
        self.traceback = traceback
        super(HTTPError, self).__init__(body, status, **options)





###############################################################################
# Plugins ######################################################################
###############################################################################

class PluginError(BottleException): pass


class JSONPlugin(object):
    name = 'json'
    api  = 2

    def __init__(self, json_dumps=json_dumps):
        self.json_dumps = json_dumps

    def apply(self, callback, route):
        dumps = self.json_dumps
        if not dumps: return callback
        def wrapper(*a, **ka):
            try:
                rv = callback(*a, **ka)
            except HTTPError:
                rv = _e()

            if isinstance(rv, dict):
                #Attempt to serialize, raises exception on failure
                json_response = dumps(rv)
                #Set content type only if serialization succesful
                response.content_type = 'application/json'
                return json_response
            elif isinstance(rv, HTTPResponse) and isinstance(rv.body, dict):
                rv.body = dumps(rv.body)
                rv.content_type = 'application/json'
            return rv

        return wrapper


class TemplatePlugin(object):
    ''' This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. '''
    name = 'template'
    api  = 2

    def apply(self, callback, route):
        conf = route.config.get('template')
        if isinstance(conf, (tuple, list)) and len(conf) == 2:
            return view(conf[0], **conf[1])(callback)
        elif isinstance(conf, str):
            return view(conf)(callback)
        else:
            return callback


#: Not a plugin, but part of the plugin API. TODO: Find a better place.
class _ImportRedirect(object):
    def __init__(self, name, impmask):
        ''' Create a virtual package that redirects imports (see PEP 302). '''
        self.name = name
        self.impmask = impmask
        self.module = sys.modules.setdefault(name, imp.new_module(name))
        self.module.__dict__.update({'__file__': __file__, '__path__': [],
                                    '__all__': [], '__loader__': self})
        sys.meta_path.append(self)

    def find_module(self, fullname, path=None):
        if '.' not in fullname: return
        packname = fullname.rsplit('.', 1)[0]
        if packname != self.name: return
        return self

    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        modname = fullname.rsplit('.', 1)[1]
        realname = self.impmask % modname
        __import__(realname)
        module = sys.modules[fullname] = sys.modules[realname]
        setattr(self.module, modname, module)
        module.__loader__ = self
        return module






###############################################################################
# Common Utilities #############################################################
###############################################################################


class MultiDict(DictMixin):
    """ This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    """

    def __init__(self, *a, **k):
        self.dict = dict((k, [v]) for (k, v) in dict(*a, **k).items())

    def __len__(self): return len(self.dict)
    def __iter__(self): return iter(self.dict)
    def __contains__(self, key): return key in self.dict
    def __delitem__(self, key): del self.dict[key]
    def __getitem__(self, key): return self.dict[key][-1]
    def __setitem__(self, key, value): self.append(key, value)
    def keys(self): return self.dict.keys()

    if py3k:
        def values(self): return (v[-1] for v in self.dict.values())
        def items(self): return ((k, v[-1]) for k, v in self.dict.items())
        def allitems(self):
            return ((k, v) for k, vl in self.dict.items() for v in vl)
        iterkeys = keys
        itervalues = values
        iteritems = items
        iterallitems = allitems

    else:
        def values(self): return [v[-1] for v in self.dict.values()]
        def items(self): return [(k, v[-1]) for k, v in self.dict.items()]
        def iterkeys(self): return self.dict.iterkeys()
        def itervalues(self): return (v[-1] for v in self.dict.itervalues())
        def iteritems(self):
            return ((k, v[-1]) for k, v in self.dict.iteritems())
        def iterallitems(self):
            return ((k, v) for k, vl in self.dict.iteritems() for v in vl)
        def allitems(self):
            return [(k, v) for k, vl in self.dict.iteritems() for v in vl]

    def get(self, key, default=None, index=-1, type=None):
        ''' Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        '''
        try:
            val = self.dict[key][index]
            return type(val) if type else val
        except Exception:
            pass
        return default

    def append(self, key, value):
        ''' Add a new value to the list of values for this key. '''
        self.dict.setdefault(key, []).append(value)

    def replace(self, key, value):
        ''' Replace the list of values with a single value. '''
        self.dict[key] = [value]

    def getall(self, key):
        ''' Return a (possibly empty) list of values for a key. '''
        return self.dict.get(key) or []

    #: Aliases for WTForms to mimic other multi-dict APIs (Django)
    getone = get
    getlist = getall


class FormsDict(MultiDict):
    ''' This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. '''

    #: Encoding used for attribute values.
    input_encoding = 'utf8'
    #: If true (default), unicode strings are first encoded with `latin1`
    #: and then decoded to match :attr:`input_encoding`.
    recode_unicode = True

    def _fix(self, s, encoding=None):
        if isinstance(s, unicode) and self.recode_unicode: # Python 3 WSGI
            return s.encode('latin1').decode(encoding or self.input_encoding)
        elif isinstance(s, bytes): # Python 2 WSGI
            return s.decode(encoding or self.input_encoding)
        else:
            return s

    def decode(self, encoding=None):
        ''' Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. '''
        copy = FormsDict()
        enc = copy.input_encoding = encoding or self.input_encoding
        copy.recode_unicode = False
        for key, value in self.allitems():
            copy.append(self._fix(key, enc), self._fix(value, enc))
        return copy

    def getunicode(self, name, default=None, encoding=None):
        ''' Return the value as a unicode string, or the default. '''
        try:
            return self._fix(self[name], encoding)
        except (UnicodeError, KeyError):
            return default

    def __getattr__(self, name, default=unicode()):
        # Without this guard, pickle generates a cryptic TypeError:
        if name.startswith('__') and name.endswith('__'):
            return super(FormsDict, self).__getattr__(name)
        return self.getunicode(name, default=default)


class HeaderDict(MultiDict):
    """ A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. """

    def __init__(self, *a, **ka):
        self.dict = {}
        if a or ka: self.update(*a, **ka)

    def __contains__(self, key): return _hkey(key) in self.dict
    def __delitem__(self, key): del self.dict[_hkey(key)]
    def __getitem__(self, key): return self.dict[_hkey(key)][-1]
    def __setitem__(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def append(self, key, value):
        self.dict.setdefault(_hkey(key), []).append(str(value))
    def replace(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def getall(self, key): return self.dict.get(_hkey(key)) or []
    def get(self, key, default=None, index=-1):
        return MultiDict.get(self, _hkey(key), default, index)
    def filter(self, names):
        for name in [_hkey(n) for n in names]:
            if name in self.dict:
                del self.dict[name]


class WSGIHeaderDict(DictMixin):
    ''' This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    '''
    #: List of keys that do not have a ``HTTP_`` prefix.
    cgikeys = ('CONTENT_TYPE', 'CONTENT_LENGTH')

    def __init__(self, environ):
        self.environ = environ

    def _ekey(self, key):
        ''' Translate header field name to CGI/WSGI environ key. '''
        key = key.replace('-','_').upper()
        if key in self.cgikeys:
            return key
        return 'HTTP_' + key

    def raw(self, key, default=None):
        ''' Return the header value as is (may be bytes or unicode). '''
        return self.environ.get(self._ekey(key), default)

    def __getitem__(self, key):
        return tonat(self.environ[self._ekey(key)], 'latin1')

    def __setitem__(self, key, value):
        raise TypeError("%s is read-only." % self.__class__)

    def __delitem__(self, key):
        raise TypeError("%s is read-only." % self.__class__)

    def __iter__(self):
        for key in self.environ:
            if key[:5] == 'HTTP_':
                yield key[5:].replace('_', '-').title()
            elif key in self.cgikeys:
                yield key.replace('_', '-').title()

    def keys(self): return [x for x in self]
    def __len__(self): return len(self.keys())
    def __contains__(self, key): return self._ekey(key) in self.environ



class ConfigDict(dict):
    ''' A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    '''
    __slots__ = ('_meta', '_on_change')

    class Namespace(DictMixin):

        def __init__(self, config, namespace):
            self._config = config
            self._prefix = namespace

        def __getitem__(self, key):
            depr('Accessing namespaces as dicts is discouraged. '
                 'Only use flat item access: '
                 'cfg["names"]["pace"]["key"] -> cfg["name.space.key"]') #0.12
            return self._config[self._prefix + '.' + key]

        def __setitem__(self, key, value):
            self._config[self._prefix + '.' + key] = value

        def __delitem__(self, key):
            del self._config[self._prefix + '.' + key]

        def __iter__(self):
            ns_prefix = self._prefix + '.'
            for key in self._config:
                ns, dot, name = key.rpartition('.')
                if ns == self._prefix and name:
                    yield name

        def keys(self): return [x for x in self]
        def __len__(self): return len(self.keys())
        def __contains__(self, key): return self._prefix + '.' + key in self._config
        def __repr__(self): return '<Config.Namespace %s.*>' % self._prefix
        def __str__(self): return '<Config.Namespace %s.*>' % self._prefix

        # Deprecated ConfigDict features
        def __getattr__(self, key):
            depr('Attribute access is deprecated.') #0.12
            if key not in self and key[0].isupper():
                self[key] = ConfigDict.Namespace(self._config, self._prefix + '.' + key)
            if key not in self and key.startswith('__'):
                raise AttributeError(key)
            return self.get(key)

        def __setattr__(self, key, value):
            if key in ('_config', '_prefix'):
                self.__dict__[key] = value
                return
            depr('Attribute assignment is deprecated.') #0.12
            if hasattr(DictMixin, key):
                raise AttributeError('Read-only attribute.')
            if key in self and self[key] and isinstance(self[key], self.__class__):
                raise AttributeError('Non-empty namespace attribute.')
            self[key] = value

        def __delattr__(self, key):
            if key in self:
                val = self.pop(key)
                if isinstance(val, self.__class__):
                    prefix = key + '.'
                    for key in self:
                        if key.startswith(prefix):
                            del self[prefix+key]

        def __call__(self, *a, **ka):
            depr('Calling ConfDict is deprecated. Use the update() method.') #0.12
            self.update(*a, **ka)
            return self

    def __init__(self, *a, **ka):
        self._meta = {}
        self._on_change = lambda name, value: None
        if a or ka:
            depr('Constructor does no longer accept parameters.') #0.12
            self.update(*a, **ka)

    def load_config(self, filename):
        ''' Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        '''
        conf = ConfigParser()
        conf.read(filename)
        for section in conf.sections():
            for key, value in conf.items(section):
                if section not in ('DEFAULT', 'bottle'):
                    key = section + '.' + key
                self[key] = value
        return self

    def load_dict(self, source, namespace='', make_namespaces=False):
        ''' Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        '''
        stack = [(namespace, source)]
        while stack:
            prefix, source = stack.pop()
            if not isinstance(source, dict):
                raise TypeError('Source is not a dict (r)' % type(key))
            for key, value in source.items():
                if not isinstance(key, basestring):
                    raise TypeError('Key is not a string (%r)' % type(key))
                full_key = prefix + '.' + key if prefix else key
                if isinstance(value, dict):
                    stack.append((full_key, value))
                    if make_namespaces:
                        self[full_key] = self.Namespace(self, full_key)
                else:
                    self[full_key] = value
        return self

    def update(self, *a, **ka):
        ''' If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` '''
        prefix = ''
        if a and isinstance(a[0], basestring):
            prefix = a[0].strip('.') + '.'
            a = a[1:]
        for key, value in dict(*a, **ka).items():
            self[prefix+key] = value

    def setdefault(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]

    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise TypeError('Key has type %r (not a string)' % type(key))

        value = self.meta_get(key, 'filter', lambda x: x)(value)
        if key in self and self[key] is value:
            return
        self._on_change(key, value)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)

    def clear(self):
        for key in self:
            del self[key]

    def meta_get(self, key, metafield, default=None):
        ''' Return the value of a meta field for a key. '''
        return self._meta.get(key, {}).get(metafield, default)

    def meta_set(self, key, metafield, value):
        ''' Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. '''
        self._meta.setdefault(key, {})[metafield] = value
        if key in self:
            self[key] = self[key]

    def meta_list(self, key):
        ''' Return an iterable of meta field names defined for a key. '''
        return self._meta.get(key, {}).keys()

    # Deprecated ConfigDict features
    def __getattr__(self, key):
        depr('Attribute access is deprecated.') #0.12
        if key not in self and key[0].isupper():
            self[key] = self.Namespace(self, key)
        if key not in self and key.startswith('__'):
            raise AttributeError(key)
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__slots__:
            return dict.__setattr__(self, key, value)
        depr('Attribute assignment is deprecated.') #0.12
        if hasattr(dict, key):
            raise AttributeError('Read-only attribute.')
        if key in self and self[key] and isinstance(self[key], self.Namespace):
            raise AttributeError('Non-empty namespace attribute.')
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            val = self.pop(key)
            if isinstance(val, self.Namespace):
                prefix = key + '.'
                for key in self:
                    if key.startswith(prefix):
                        del self[prefix+key]

    def __call__(self, *a, **ka):
        depr('Calling ConfDict is deprecated. Use the update() method.') #0.12
        self.update(*a, **ka)
        return self



class AppStack(list):
    """ A stack-like list. Calling it returns the head of the stack. """

    def __call__(self):
        """ Return the current default application. """
        return self[-1]

    def push(self, value=None):
        """ Add a new :class:`Bottle` instance to the stack """
        if not isinstance(value, Bottle):
            value = Bottle()
        self.append(value)
        return value


class WSGIFileWrapper(object):

    def __init__(self, fp, buffer_size=1024*64):
        self.fp, self.buffer_size = fp, buffer_size
        for attr in ('fileno', 'close', 'read', 'readlines', 'tell', 'seek'):
            if hasattr(fp, attr): setattr(self, attr, getattr(fp, attr))

    def __iter__(self):
        buff, read = self.buffer_size, self.read
        while True:
            part = read(buff)
            if not part: return
            yield part


class _closeiter(object):
    ''' This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). '''

    def __init__(self, iterator, close=None):
        self.iterator = iterator
        self.close_callbacks = makelist(close)

    def __iter__(self):
        return iter(self.iterator)

    def close(self):
        for func in self.close_callbacks:
            func()


class ResourceManager(object):
    ''' This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    '''

    def __init__(self, base='./', opener=open, cachemode='all'):
        self.opener = open
        self.base = base
        self.cachemode = cachemode

        #: A list of search paths. See :meth:`add_path` for details.
        self.path = []
        #: A cache for resolved paths. ``res.cache.clear()`` clears the cache.
        self.cache = {}

    def add_path(self, path, base=None, index=None, create=False):
        ''' Add a new path to the list of search paths. Return False if the
            path does not exist.

            :param path: The new search path. Relative paths are turned into
                an absolute and normalized form. If the path looks like a file
                (not ending in `/`), the filename is stripped off.
            :param base: Path used to absolutize relative search paths.
                Defaults to :attr:`base` which defaults to ``os.getcwd()``.
            :param index: Position within the list of search paths. Defaults
                to last index (appends to the list).

            The `base` parameter makes it easy to reference files installed
            along with a python module or package::

                res.add_path('./resources/', __file__)
        '''
        base = os.path.abspath(os.path.dirname(base or self.base))
        path = os.path.abspath(os.path.join(base, os.path.dirname(path)))
        path += os.sep
        if path in self.path:
            self.path.remove(path)
        if create and not os.path.isdir(path):
            os.makedirs(path)
        if index is None:
            self.path.append(path)
        else:
            self.path.insert(index, path)
        self.cache.clear()
        return os.path.exists(path)

    def __iter__(self):
        ''' Iterate over all existing files in all registered paths. '''
        search = self.path[:]
        while search:
            path = search.pop()
            if not os.path.isdir(path): continue
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isdir(full): search.append(full)
                else: yield full

    def lookup(self, name):
        ''' Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. '''
        if name not in self.cache or DEBUG:
            for path in self.path:
                fpath = os.path.join(path, name)
                if os.path.isfile(fpath):
                    if self.cachemode in ('all', 'found'):
                        self.cache[name] = fpath
                    return fpath
            if self.cachemode == 'all':
                self.cache[name] = None
        return self.cache[name]

    def open(self, name, mode='r', *args, **kwargs):
        ''' Find a resource and return a file object, or raise IOError. '''
        fname = self.lookup(name)
        if not fname: raise IOError("Resource %r not found." % name)
        return self.opener(fname, mode=mode, *args, **kwargs)


class FileUpload(object):

    def __init__(self, fileobj, name, filename, headers=None):
        ''' Wrapper for file uploads. '''
        #: Open file(-like) object (BytesIO buffer or temporary file)
        self.file = fileobj
        #: Name of the upload form field
        self.name = name
        #: Raw filename as sent by the client (may contain unsafe characters)
        self.raw_filename = filename
        #: A :class:`HeaderDict` with additional headers (e.g. content-type)
        self.headers = HeaderDict(headers) if headers else HeaderDict()

    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int, default=-1)

    @cached_property
    def filename(self):
        ''' Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        '''
        fname = self.raw_filename
        if not isinstance(fname, unicode):
            fname = fname.decode('utf8', 'ignore')
        fname = normalize('NFKD', fname).encode('ASCII', 'ignore').decode('ASCII')
        fname = os.path.basename(fname.replace('\\', os.path.sep))
        fname = re.sub(r'[^a-zA-Z0-9-_.\s]', '', fname).strip()
        fname = re.sub(r'[-\s]+', '-', fname).strip('.-')
        return fname[:255] or 'empty'

    def _copy_file(self, fp, chunk_size=2**16):
        read, write, offset = self.file.read, fp.write, self.file.tell()
        while 1:
            buf = read(chunk_size)
            if not buf: break
            write(buf)
        self.file.seek(offset)

    def save(self, destination, overwrite=False, chunk_size=2**16):
        ''' Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        '''
        if isinstance(destination, basestring): # Except file-likes here
            if os.path.isdir(destination):
                destination = os.path.join(destination, self.filename)
            if not overwrite and os.path.exists(destination):
                raise IOError('File exists.')
            with open(destination, 'wb') as fp:
                self._copy_file(fp, chunk_size)
        else:
            self._copy_file(destination, chunk_size)






###############################################################################
# Application Helper ###########################################################
###############################################################################


def abort(code=500, text='Unknown Error.'):
    """ Aborts execution and causes a HTTP error. """
    raise HTTPError(code, text)


def redirect(url, code=None):
    """ Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. """
    if not code:
        code = 303 if request.get('SERVER_PROTOCOL') == "HTTP/1.1" else 302
    res = response.copy(cls=HTTPResponse)
    res.status = code
    res.body = ""
    res.set_header('Location', urljoin(request.url, url))
    raise res


def _file_iter_range(fp, offset, bytes, maxread=1024*1024):
    ''' Yield chunks from a range in a file. No chunk is bigger than maxread.'''
    fp.seek(offset)
    while bytes > 0:
        part = fp.read(min(bytes, maxread))
        if not part: break
        bytes -= len(part)
        yield part


def static_file(filename, root, mimetype='auto', download=False, charset='UTF-8'):
    """ Open a file in a safe way and return :exc:`HTTPResponse` with status
        code 200, 305, 403 or 404. The ``Content-Type``, ``Content-Encoding``,
        ``Content-Length`` and ``Last-Modified`` headers are set if possible.
        Special support for ``If-Modified-Since``, ``Range`` and ``HEAD``
        requests.

        :param filename: Name or path of the file to send.
        :param root: Root path for file lookups. Should be an absolute directory
            path.
        :param mimetype: Defines the content-type header (default: guess from
            file extension)
        :param download: If True, ask the browser to open a `Save as...` dialog
            instead of opening the file with the associated program. You can
            specify a custom filename as a string. If not specified, the
            original filename is used (default: False).
        :param charset: The charset to use for files with a ``text/*``
            mime-type. (default: UTF-8)
    """

    root = os.path.abspath(root) + os.sep
    filename = os.path.abspath(os.path.join(root, filename.strip('/\\')))
    headers = dict()

    if not filename.startswith(root):
        return HTTPError(403, "Access denied.")
    if not os.path.exists(filename) or not os.path.isfile(filename):
        return HTTPError(404, "File does not exist.")
    if not os.access(filename, os.R_OK):
        return HTTPError(403, "You do not have permission to access this file.")

    if mimetype == 'auto':
        mimetype, encoding = mimetypes.guess_type(filename)
        if encoding: headers['Content-Encoding'] = encoding

    if mimetype:
        if mimetype[:5] == 'text/' and charset and 'charset' not in mimetype:
            mimetype += '; charset=%s' % charset
        headers['Content-Type'] = mimetype

    if download:
        download = os.path.basename(filename if download == True else download)
        headers['Content-Disposition'] = 'attachment; filename="%s"' % download

    stats = os.stat(filename)
    headers['Content-Length'] = clen = stats.st_size
    lm = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stats.st_mtime))
    headers['Last-Modified'] = lm

    ims = request.environ.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims = parse_date(ims.split(";")[0].strip())
    if ims is not None and ims >= int(stats.st_mtime):
        headers['Date'] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        return HTTPResponse(status=304, **headers)

    body = '' if request.method == 'HEAD' else open(filename, 'rb')

    headers["Accept-Ranges"] = "bytes"
    ranges = request.environ.get('HTTP_RANGE')
    if 'HTTP_RANGE' in request.environ:
        ranges = list(parse_range_header(request.environ['HTTP_RANGE'], clen))
        if not ranges:
            return HTTPError(416, "Requested Range Not Satisfiable")
        offset, end = ranges[0]
        headers["Content-Range"] = "bytes %d-%d/%d" % (offset, end-1, clen)
        headers["Content-Length"] = str(end-offset)
        if body: body = _file_iter_range(body, offset, end-offset)
        return HTTPResponse(body, status=206, **headers)
    return HTTPResponse(body, **headers)






###############################################################################
# HTTP Utilities and MISC (TODO) ###############################################
###############################################################################


def debug(mode=True):
    """ Change the debug level.
    There is only one debug level supported at the moment."""
    global DEBUG
    if mode: warnings.simplefilter('default')
    DEBUG = bool(mode)

def http_date(value):
    if isinstance(value, (datedate, datetime)):
        value = value.utctimetuple()
    elif isinstance(value, (int, float)):
        value = time.gmtime(value)
    if not isinstance(value, basestring):
        value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
    return value

def parse_date(ims):
    """ Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. """
    try:
        ts = email.utils.parsedate_tz(ims)
        return time.mktime(ts[:8] + (0,)) - (ts[9] or 0) - time.timezone
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def parse_auth(header):
    """ Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or None"""
    try:
        method, data = header.split(None, 1)
        if method.lower() == 'basic':
            user, pwd = touni(base64.b64decode(tob(data))).split(':',1)
            return user, pwd
    except (KeyError, ValueError):
        return None

def parse_range_header(header, maxlen=0):
    ''' Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.'''
    if not header or header[:6] != 'bytes=': return
    ranges = [r.split('-', 1) for r in header[6:].split(',') if '-' in r]
    for start, end in ranges:
        try:
            if not start:  # bytes=-100    -> last 100 bytes
                start, end = max(0, maxlen-int(end)), maxlen
            elif not end:  # bytes=100-    -> all but the first 99 bytes
                start, end = int(start), maxlen
            else:          # bytes=100-200 -> bytes 100-200 (inclusive)
                start, end = int(start), min(int(end)+1, maxlen)
            if 0 <= start < end <= maxlen:
                yield start, end
        except ValueError:
            pass

def _parse_qsl(qs):
    r = []
    for pair in qs.replace(';','&').split('&'):
        if not pair: continue
        nv = pair.split('=', 1)
        if len(nv) != 2: nv.append('')
        key = urlunquote(nv[0].replace('+', ' '))
        value = urlunquote(nv[1].replace('+', ' '))
        r.append((key, value))
    return r

def _lscmp(a, b):
    ''' Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. '''
    return not sum(0 if x==y else 1 for x, y in zip(a, b)) and len(a) == len(b)


def cookie_encode(data, key):
    ''' Encode and sign a pickle-able object. Return a (byte) string '''
    msg = base64.b64encode(pickle.dumps(data, -1))
    sig = base64.b64encode(hmac.new(tob(key), msg).digest())
    return tob('!') + sig + tob('?') + msg


def cookie_decode(data, key):
    ''' Verify and decode an encoded string. Return an object or None.'''
    data = tob(data)
    if cookie_is_encoded(data):
        sig, msg = data.split(tob('?'), 1)
        if _lscmp(sig[1:], base64.b64encode(hmac.new(tob(key), msg).digest())):
            return pickle.loads(base64.b64decode(msg))
    return None


def cookie_is_encoded(data):
    ''' Return True if the argument looks like a encoded cookie.'''
    return bool(data.startswith(tob('!')) and tob('?') in data)


def html_escape(string):
    ''' Escape HTML special characters ``&<>`` and quotes ``'"``. '''
    return string.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')\
                 .replace('"','&quot;').replace("'",'&#039;')


def html_quote(string):
    ''' Escape and quote a string to be used as an HTTP attribute.'''
    return '"%s"' % html_escape(string).replace('\n','&#10;')\
                    .replace('\r','&#13;').replace('\t','&#9;')


def yieldroutes(func):
    """ Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    """
    path = '/' + func.__name__.replace('__','/').lstrip('/')
    spec = getargspec(func)
    argc = len(spec[0]) - len(spec[3] or [])
    path += ('/<%s>' * argc) % tuple(spec[0][:argc])
    yield path
    for arg in spec[0][argc:]:
        path += '/<%s>' % arg
        yield path


def path_shift(script_name, path_info, shift=1):
    ''' Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    '''
    if shift == 0: return script_name, path_info
    pathlist = path_info.strip('/').split('/')
    scriptlist = script_name.strip('/').split('/')
    if pathlist and pathlist[0] == '': pathlist = []
    if scriptlist and scriptlist[0] == '': scriptlist = []
    if shift > 0 and shift <= len(pathlist):
        moved = pathlist[:shift]
        scriptlist = scriptlist + moved
        pathlist = pathlist[shift:]
    elif shift < 0 and shift >= -len(scriptlist):
        moved = scriptlist[shift:]
        pathlist = moved + pathlist
        scriptlist = scriptlist[:shift]
    else:
        empty = 'SCRIPT_NAME' if shift < 0 else 'PATH_INFO'
        raise AssertionError("Cannot shift. Nothing left from %s" % empty)
    new_script_name = '/' + '/'.join(scriptlist)
    new_path_info = '/' + '/'.join(pathlist)
    if path_info.endswith('/') and pathlist: new_path_info += '/'
    return new_script_name, new_path_info


def auth_basic(check, realm="private", text="Access denied"):
    ''' Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. '''
    def decorator(func):
        def wrapper(*a, **ka):
            user, password = request.auth or (None, None)
            if user is None or not check(user, password):
                err = HTTPError(401, text)
                err.add_header('WWW-Authenticate', 'Basic realm="%s"' % realm)
                return err
            return func(*a, **ka)
        return wrapper
    return decorator


# Shortcuts for common Bottle methods.
# They all refer to the current default application.

def make_default_app_wrapper(name):
    ''' Return a callable that relays calls to the current default app. '''
    @functools.wraps(getattr(Bottle, name))
    def wrapper(*a, **ka):
        return getattr(app(), name)(*a, **ka)
    return wrapper

route     = make_default_app_wrapper('route')
get       = make_default_app_wrapper('get')
post      = make_default_app_wrapper('post')
put       = make_default_app_wrapper('put')
delete    = make_default_app_wrapper('delete')
error     = make_default_app_wrapper('error')
mount     = make_default_app_wrapper('mount')
hook      = make_default_app_wrapper('hook')
install   = make_default_app_wrapper('install')
uninstall = make_default_app_wrapper('uninstall')
url       = make_default_app_wrapper('get_url')







###############################################################################
# Server Adapter ###############################################################
###############################################################################


class ServerAdapter(object):
    quiet = False
    def __init__(self, host='127.0.0.1', port=8080, **options):
        self.options = options
        self.host = host
        self.port = int(port)

    def run(self, handler): # pragma: no cover
        pass

    def __repr__(self):
        args = ', '.join(['%s=%s'%(k,repr(v)) for k, v in self.options.items()])
        return "%s(%s)" % (self.__class__.__name__, args)


class CGIServer(ServerAdapter):
    quiet = True
    def run(self, handler): # pragma: no cover
        from wsgiref.handlers import CGIHandler
        def fixed_environ(environ, start_response):
            environ.setdefault('PATH_INFO', '')
            return handler(environ, start_response)
        CGIHandler().run(fixed_environ)


class FlupFCGIServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        import flup.server.fcgi
        self.options.setdefault('bindAddress', (self.host, self.port))
        flup.server.fcgi.WSGIServer(handler, **self.options).run()


class WSGIRefServer(ServerAdapter):
    def run(self, app): # pragma: no cover
        from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
        from wsgiref.simple_server import make_server
        import socket

        class FixedHandler(WSGIRequestHandler):
            def address_string(self): # Prevent reverse DNS lookups please.
                return self.client_address[0]
            def log_request(*args, **kw):
                if not self.quiet:
                    return WSGIRequestHandler.log_request(*args, **kw)

        handler_cls = self.options.get('handler_class', FixedHandler)
        server_cls  = self.options.get('server_class', WSGIServer)

        if ':' in self.host: # Fix wsgiref for IPv6 addresses.
            if getattr(server_cls, 'address_family') == socket.AF_INET:
                class server_cls(server_cls):
                    address_family = socket.AF_INET6

        srv = make_server(self.host, self.port, app, server_cls, handler_cls)
        srv.serve_forever()


class CherryPyServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        from cherrypy import wsgiserver
        self.options['bind_addr'] = (self.host, self.port)
        self.options['wsgi_app'] = handler

        certfile = self.options.get('certfile')
        if certfile:
            del self.options['certfile']
        keyfile = self.options.get('keyfile')
        if keyfile:
            del self.options['keyfile']

        server = wsgiserver.CherryPyWSGIServer(**self.options)
        if certfile:
            server.ssl_certificate = certfile
        if keyfile:
            server.ssl_private_key = keyfile

        try:
            server.start()
        finally:
            server.stop()


class WaitressServer(ServerAdapter):
    def run(self, handler):
        from waitress import serve
        serve(handler, host=self.host, port=self.port)


class PasteServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        from paste import httpserver
        from paste.translogger import TransLogger
        handler = TransLogger(handler, setup_console_handler=(not self.quiet))
        httpserver.serve(handler, host=self.host, port=str(self.port),
                         **self.options)


class MeinheldServer(ServerAdapter):
    def run(self, handler):
        from meinheld import server
        server.listen((self.host, self.port))
        server.run(handler)


class FapwsServer(ServerAdapter):
    """ Extremely fast webserver using libev. See http://www.fapws.org/ """
    def run(self, handler): # pragma: no cover
        import fapws._evwsgi as evwsgi
        from fapws import base, config
        port = self.port
        if float(config.SERVER_IDENT[-2:]) > 0.4:
            # fapws3 silently changed its API in 0.5
            port = str(port)
        evwsgi.start(self.host, port)
        # fapws3 never releases the GIL. Complain upstream. I tried. No luck.
        if 'BOTTLE_CHILD' in os.environ and not self.quiet:
            _stderr("WARNING: Auto-reloading does not work with Fapws3.\n")
            _stderr("         (Fapws3 breaks python thread support)\n")
        evwsgi.set_base_module(base)
        def app(environ, start_response):
            environ['wsgi.multiprocess'] = False
            return handler(environ, start_response)
        evwsgi.wsgi_cb(('', app))
        evwsgi.run()


class TornadoServer(ServerAdapter):
    """ The super hyped asynchronous server by facebook. Untested. """
    def run(self, handler): # pragma: no cover
        import tornado.wsgi, tornado.httpserver, tornado.ioloop
        container = tornado.wsgi.WSGIContainer(handler)
        server = tornado.httpserver.HTTPServer(container)
        server.listen(port=self.port,address=self.host)
        tornado.ioloop.IOLoop.instance().start()


class AppEngineServer(ServerAdapter):
    """ Adapter for Google App Engine. """
    quiet = True
    def run(self, handler):
        from google.appengine.ext.webapp import util
        # A main() function in the handler script enables 'App Caching'.
        # Lets makes sure it is there. This _really_ improves performance.
        module = sys.modules.get('__main__')
        if module and not hasattr(module, 'main'):
            module.main = lambda: util.run_wsgi_app(handler)
        util.run_wsgi_app(handler)


class TwistedServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from twisted.web import server, wsgi
        from twisted.python.threadpool import ThreadPool
        from twisted.internet import reactor
        thread_pool = ThreadPool()
        thread_pool.start()
        reactor.addSystemEventTrigger('after', 'shutdown', thread_pool.stop)
        factory = server.Site(wsgi.WSGIResource(reactor, thread_pool, handler))
        reactor.listenTCP(self.port, factory, interface=self.host)
        reactor.run()


class DieselServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from diesel.protocols.wsgi import WSGIApplication
        app = WSGIApplication(handler, port=self.port)
        app.run()


class GeventServer(ServerAdapter):
    """ Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    """
    def run(self, handler):
        from gevent import wsgi, pywsgi, local
        if not isinstance(threading.local(), local.local):
            msg = "Bottle requires gevent.monkey.patch_all() (before import)"
            raise RuntimeError(msg)
        if not self.options.pop('fast', None): wsgi = pywsgi
        self.options['log'] = None if self.quiet else 'default'
        address = (self.host, self.port)
        server = wsgi.WSGIServer(address, handler, **self.options)
        if 'BOTTLE_CHILD' in os.environ:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: server.stop())
        server.serve_forever()


class GeventSocketIOServer(ServerAdapter):
    def run(self,handler):
        from socketio import server
        address = (self.host, self.port)
        server.SocketIOServer(address, handler, **self.options).serve_forever()


class GunicornServer(ServerAdapter):
    """ Untested. See http://gunicorn.org/configure.html for options. """
    def run(self, handler):
        from gunicorn.app.base import Application

        config = {'bind': "%s:%d" % (self.host, int(self.port))}
        config.update(self.options)

        class GunicornApplication(Application):
            def init(self, parser, opts, args):
                return config

            def load(self):
                return handler

        GunicornApplication().run()


class EventletServer(ServerAdapter):
    """ Untested """
    def run(self, handler):
        from eventlet import wsgi, listen
        try:
            wsgi.server(listen((self.host, self.port)), handler,
                        log_output=(not self.quiet))
        except TypeError:
            # Fallback, if we have old version of eventlet
            wsgi.server(listen((self.host, self.port)), handler)


class RocketServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from rocket import Rocket
        server = Rocket((self.host, self.port), 'wsgi', { 'wsgi_app' : handler })
        server.start()


class BjoernServer(ServerAdapter):
    """ Fast server written in C: https://github.com/jonashaag/bjoern """
    def run(self, handler):
        from bjoern import run
        run(handler, self.host, self.port)


class AutoServer(ServerAdapter):
    """ Untested. """
    adapters = [WaitressServer, PasteServer, TwistedServer, CherryPyServer, WSGIRefServer]
    def run(self, handler):
        for sa in self.adapters:
            try:
                return sa(self.host, self.port, **self.options).run(handler)
            except ImportError:
                pass

server_names = {
    'cgi': CGIServer,
    'flup': FlupFCGIServer,
    'wsgiref': WSGIRefServer,
    'waitress': WaitressServer,
    'cherrypy': CherryPyServer,
    'paste': PasteServer,
    'fapws3': FapwsServer,
    'tornado': TornadoServer,
    'gae': AppEngineServer,
    'twisted': TwistedServer,
    'diesel': DieselServer,
    'meinheld': MeinheldServer,
    'gunicorn': GunicornServer,
    'eventlet': EventletServer,
    'gevent': GeventServer,
    'geventSocketIO':GeventSocketIOServer,
    'rocket': RocketServer,
    'bjoern' : BjoernServer,
    'auto': AutoServer,
}






###############################################################################
# Application Control ##########################################################
###############################################################################


def load(target, **namespace):
    """ Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    """
    module, target = target.split(":", 1) if ':' in target else (target, None)
    if module not in sys.modules: __import__(module)
    if not target: return sys.modules[module]
    if target.isalnum(): return getattr(sys.modules[module], target)
    package_name = module.split('.')[0]
    namespace[package_name] = sys.modules[package_name]
    return eval('%s.%s' % (module, target), namespace)


def load_app(target):
    """ Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. """
    global NORUN; NORUN, nr_old = True, NORUN
    try:
        tmp = default_app.push() # Create a new "default application"
        rv = load(target) # Import the target module
        return rv if callable(rv) else tmp
    finally:
        default_app.remove(tmp) # Remove the temporary added default application
        NORUN = nr_old

_debug = debug
def run(app=None, server='wsgiref', host='127.0.0.1', port=8080,
        interval=1, reloader=False, quiet=False, plugins=None,
        debug=None, **kargs):
    """ Start a server instance. This method blocks until the server terminates.

        :param app: WSGI application or target string supported by
               :func:`load_app`. (default: :func:`default_app`)
        :param server: Server adapter to use. See :data:`server_names` keys
               for valid names or pass a :class:`ServerAdapter` subclass.
               (default: `wsgiref`)
        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on
               all interfaces including the external one. (default: 127.0.0.1)
        :param port: Server port to bind to. Values below 1024 require root
               privileges. (default: 8080)
        :param reloader: Start auto-reloading server? (default: False)
        :param interval: Auto-reloader interval in seconds (default: 1)
        :param quiet: Suppress output to stdout and stderr? (default: False)
        :param options: Options passed to the server adapter.
     """
    if NORUN: return
    if reloader and not os.environ.get('BOTTLE_CHILD'):
        try:
            lockfile = None
            fd, lockfile = tempfile.mkstemp(prefix='bottle.', suffix='.lock')
            os.close(fd) # We only need this file to exist. We never write to it
            while os.path.exists(lockfile):
                args = [sys.executable] + sys.argv
                environ = os.environ.copy()
                environ['BOTTLE_CHILD'] = 'true'
                environ['BOTTLE_LOCKFILE'] = lockfile
                p = subprocess.Popen(args, env=environ)
                while p.poll() is None: # Busy wait...
                    os.utime(lockfile, None) # I am alive!
                    time.sleep(interval)
                if p.poll() != 3:
                    if os.path.exists(lockfile): os.unlink(lockfile)
                    sys.exit(p.poll())
        except KeyboardInterrupt:
            pass
        finally:
            if os.path.exists(lockfile):
                os.unlink(lockfile)
        return

    try:
        if debug is not None: _debug(debug)
        app = app or default_app()
        if isinstance(app, basestring):
            app = load_app(app)
        if not callable(app):
            raise ValueError("Application is not callable: %r" % app)

        for plugin in plugins or []:
            app.install(plugin)

        if server in server_names:
            server = server_names.get(server)
        if isinstance(server, basestring):
            server = load(server)
        if isinstance(server, type):
            server = server(host=host, port=port, **kargs)
        if not isinstance(server, ServerAdapter):
            raise ValueError("Unknown or unsupported server: %r" % server)

        server.quiet = server.quiet or quiet
        if not server.quiet:
            _stderr("Bottle v%s server starting up (using %s)...\n" % (__version__, repr(server)))
            _stderr("Listening on http://%s:%d/\n" % (server.host, server.port))
            _stderr("Hit Ctrl-C to quit.\n\n")

        if reloader:
            lockfile = os.environ.get('BOTTLE_LOCKFILE')
            bgcheck = FileCheckerThread(lockfile, interval)
            with bgcheck:
                server.run(app)
            if bgcheck.status == 'reload':
                sys.exit(3)
        else:
            server.run(app)
    except KeyboardInterrupt:
        pass
    except (SystemExit, MemoryError):
        raise
    except:
        if not reloader: raise
        if not getattr(server, 'quiet', quiet):
            print_exc()
        time.sleep(interval)
        sys.exit(3)



class FileCheckerThread(threading.Thread):
    ''' Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. '''

    def __init__(self, lockfile, interval):
        threading.Thread.__init__(self)
        self.lockfile, self.interval = lockfile, interval
        #: Is one of 'reload', 'error' or 'exit'
        self.status = None

    def run(self):
        exists = os.path.exists
        mtime = lambda path: os.stat(path).st_mtime
        files = dict()

        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', '')
            if path[-4:] in ('.pyo', '.pyc'): path = path[:-1]
            if path and exists(path): files[path] = mtime(path)

        while not self.status:
            if not exists(self.lockfile)\
            or mtime(self.lockfile) < time.time() - self.interval - 5:
                self.status = 'error'
                thread.interrupt_main()
            for path, lmtime in list(files.items()):
                if not exists(path) or mtime(path) > lmtime:
                    self.status = 'reload'
                    thread.interrupt_main()
                    break
            time.sleep(self.interval)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.status: self.status = 'exit' # silent exit
        self.join()
        return exc_type is not None and issubclass(exc_type, KeyboardInterrupt)





###############################################################################
# Template Adapters ############################################################
###############################################################################


class TemplateError(HTTPError):
    def __init__(self, message):
        HTTPError.__init__(self, 500, message)


class BaseTemplate(object):
    """ Base class and minimal API for template adapters """
    extensions = ['tpl','html','thtml','stpl']
    settings = {} #used in prepare()
    defaults = {} #used in render()

    def __init__(self, source=None, name=None, lookup=[], encoding='utf8', **settings):
        """ Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        """
        self.name = name
        self.source = source.read() if hasattr(source, 'read') else source
        self.filename = source.filename if hasattr(source, 'filename') else None
        self.lookup = [os.path.abspath(x) for x in lookup]
        self.encoding = encoding
        self.settings = self.settings.copy() # Copy from class variable
        self.settings.update(settings) # Apply
        if not self.source and self.name:
            self.filename = self.search(self.name, self.lookup)
            if not self.filename:
                raise TemplateError('Template %s not found.' % repr(name))
        if not self.source and not self.filename:
            raise TemplateError('No template specified.')
        self.prepare(**self.settings)

    @classmethod
    def search(cls, name, lookup=[]):
        """ Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. """
        if not lookup:
            depr('The template lookup path list should not be empty.') #0.12
            lookup = ['.']

        if os.path.isabs(name) and os.path.isfile(name):
            depr('Absolute template path names are deprecated.') #0.12
            return os.path.abspath(name)

        for spath in lookup:
            spath = os.path.abspath(spath) + os.sep
            fname = os.path.abspath(os.path.join(spath, name))
            if not fname.startswith(spath): continue
            if os.path.isfile(fname): return fname
            for ext in cls.extensions:
                if os.path.isfile('%s.%s' % (fname, ext)):
                    return '%s.%s' % (fname, ext)

    @classmethod
    def global_config(cls, key, *args):
        ''' This reads or sets the global settings stored in class.settings. '''
        if args:
            cls.settings = cls.settings.copy() # Make settings local to class
            cls.settings[key] = args[0]
        else:
            return cls.settings[key]

    def prepare(self, **options):
        """ Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        """
        raise NotImplementedError

    def render(self, *args, **kwargs):
        """ Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        """
        raise NotImplementedError


class MakoTemplate(BaseTemplate):
    def prepare(self, **options):
        from mako.template import Template
        from mako.lookup import TemplateLookup
        options.update({'input_encoding':self.encoding})
        options.setdefault('format_exceptions', bool(DEBUG))
        lookup = TemplateLookup(directories=self.lookup, **options)
        if self.source:
            self.tpl = Template(self.source, lookup=lookup, **options)
        else:
            self.tpl = Template(uri=self.name, filename=self.filename, lookup=lookup, **options)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)


class CheetahTemplate(BaseTemplate):
    def prepare(self, **options):
        from Cheetah.Template import Template
        self.context = threading.local()
        self.context.vars = {}
        options['searchList'] = [self.context.vars]
        if self.source:
            self.tpl = Template(source=self.source, **options)
        else:
            self.tpl = Template(file=self.filename, **options)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        self.context.vars.update(self.defaults)
        self.context.vars.update(kwargs)
        out = str(self.tpl)
        self.context.vars.clear()
        return out


class Jinja2Template(BaseTemplate):
    def prepare(self, filters=None, tests=None, globals={}, **kwargs):
        from jinja2 import Environment, FunctionLoader
        if 'prefix' in kwargs: # TODO: to be removed after a while
            raise RuntimeError('The keyword argument `prefix` has been removed. '
                'Use the full jinja2 environment name line_statement_prefix instead.')
        self.env = Environment(loader=FunctionLoader(self.loader), **kwargs)
        if filters: self.env.filters.update(filters)
        if tests: self.env.tests.update(tests)
        if globals: self.env.globals.update(globals)
        if self.source:
            self.tpl = self.env.from_string(self.source)
        else:
            self.tpl = self.env.get_template(self.filename)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)

    def loader(self, name):
        fname = self.search(name, self.lookup)
        if not fname: return
        with open(fname, "rb") as f:
            return f.read().decode(self.encoding)


class SimpleTemplate(BaseTemplate):

    def prepare(self, escape_func=html_escape, noescape=False, syntax=None, **ka):
        self.cache = {}
        enc = self.encoding
        self._str = lambda x: touni(x, enc)
        self._escape = lambda x: escape_func(touni(x, enc))
        self.syntax = syntax
        if noescape:
            self._str, self._escape = self._escape, self._str

    @cached_property
    def co(self):
        return compile(self.code, self.filename or '<string>', 'exec')

    @cached_property
    def code(self):
        source = self.source
        if not source:
            with open(self.filename, 'rb') as f:
                source = f.read()
        try:
            source, encoding = touni(source), 'utf8'
        except UnicodeError:
            depr('Template encodings other than utf8 are no longer supported.') #0.11
            source, encoding = touni(source, 'latin1'), 'latin1'
        parser = StplParser(source, encoding=encoding, syntax=self.syntax)
        code = parser.translate()
        self.encoding = parser.encoding
        return code

    def _rebase(self, _env, _name=None, **kwargs):
        if _name is None:
            depr('Rebase function called without arguments.'
                 ' You were probably looking for {{base}}?', True) #0.12
        _env['_rebase'] = (_name, kwargs)

    def _include(self, _env, _name=None, **kwargs):
        if _name is None:
            depr('Rebase function called without arguments.'
                 ' You were probably looking for {{base}}?', True) #0.12
        env = _env.copy()
        env.update(kwargs)
        if _name not in self.cache:
            self.cache[_name] = self.__class__(name=_name, lookup=self.lookup)
        return self.cache[_name].execute(env['_stdout'], env)

    def execute(self, _stdout, kwargs):
        env = self.defaults.copy()
        env.update(kwargs)
        env.update({'_stdout': _stdout, '_printlist': _stdout.extend,
            'include': functools.partial(self._include, env),
            'rebase': functools.partial(self._rebase, env), '_rebase': None,
            '_str': self._str, '_escape': self._escape, 'get': env.get,
            'setdefault': env.setdefault, 'defined': env.__contains__ })
        eval(self.co, env)
        if env.get('_rebase'):
            subtpl, rargs = env.pop('_rebase')
            rargs['base'] = ''.join(_stdout) #copy stdout
            del _stdout[:] # clear stdout
            return self._include(env, subtpl, **rargs)
        return env

    def render(self, *args, **kwargs):
        """ Render the template using keyword arguments as local variables. """
        env = {}; stdout = []
        for dictarg in args: env.update(dictarg)
        env.update(kwargs)
        self.execute(stdout, env)
        return ''.join(stdout)


class StplSyntaxError(TemplateError): pass


class StplParser(object):
    ''' Parser for stpl templates. '''
    _re_cache = {} #: Cache for compiled re patterns
    # This huge pile of voodoo magic splits python code into 8 different tokens.
    # 1: All kinds of python strings (trust me, it works)
    _re_tok = '((?m)[urbURB]?(?:\'\'(?!\')|""(?!")|\'{6}|"{6}' \
               '|\'(?:[^\\\\\']|\\\\.)+?\'|"(?:[^\\\\"]|\\\\.)+?"' \
               '|\'{3}(?:[^\\\\]|\\\\.|\\n)+?\'{3}' \
               '|"{3}(?:[^\\\\]|\\\\.|\\n)+?"{3}))'
    _re_inl = _re_tok.replace('|\\n','') # We re-use this string pattern later
    # 2: Comments (until end of line, but not the newline itself)
    _re_tok += '|(#.*)'
    # 3,4: Open and close grouping tokens
    _re_tok += '|([\[\{\(])'
    _re_tok += '|([\]\}\)])'
    # 5,6: Keywords that start or continue a python block (only start of line)
    _re_tok += '|^([ \\t]*(?:if|for|while|with|try|def|class)\\b)' \
               '|^([ \\t]*(?:elif|else|except|finally)\\b)'
    # 7: Our special 'end' keyword (but only if it stands alone)
    _re_tok += '|((?:^|;)[ \\t]*end[ \\t]*(?=(?:%(block_close)s[ \\t]*)?\\r?$|;|#))'
    # 8: A customizable end-of-code-block template token (only end of line)
    _re_tok += '|(%(block_close)s[ \\t]*(?=$))'
    # 9: And finally, a single newline. The 10th token is 'everything else'
    _re_tok += '|(\\r?\\n)'

    # Match the start tokens of code areas in a template
    _re_split = '(?m)^[ \t]*(\\\\?)((%(line_start)s)|(%(block_start)s))(%%?)'
    # Match inline statements (may contain python strings)
    _re_inl = '%%(inline_start)s((?:%s|[^\'"\n]*?)+)%%(inline_end)s' % _re_inl

    default_syntax = '<% %> % {{ }}'

    def __init__(self, source, syntax=None, encoding='utf8'):
        self.source, self.encoding = touni(source, encoding), encoding
        self.set_syntax(syntax or self.default_syntax)
        self.code_buffer, self.text_buffer = [], []
        self.lineno, self.offset = 1, 0
        self.indent, self.indent_mod = 0, 0
        self.paren_depth = 0

    def get_syntax(self):
        ''' Tokens as a space separated string (default: <% %> % {{ }}) '''
        return self._syntax

    def set_syntax(self, syntax):
        self._syntax = syntax
        self._tokens = syntax.split()
        if not syntax in self._re_cache:
            names = 'block_start block_close line_start inline_start inline_end'
            etokens = map(re.escape, self._tokens)
            pattern_vars = dict(zip(names.split(), etokens))
            patterns = (self._re_split, self._re_tok, self._re_inl)
            patterns = [re.compile(p%pattern_vars) for p in patterns]
            self._re_cache[syntax] = patterns
        self.re_split, self.re_tok, self.re_inl = self._re_cache[syntax]

    syntax = property(get_syntax, set_syntax)

    def translate(self):
        if self.offset: raise RuntimeError('Parser is a one time instance.')
        while True:
            m = self.re_split.search(self.source[self.offset:])
            if m:
                text = self.source[self.offset:self.offset+m.start()]
                self.text_buffer.append(text)
                self.offset += m.end()
                if m.group(1): # New escape syntax
                    line, sep, _ = self.source[self.offset:].partition('\n')
                    self.text_buffer.append(m.group(2)+m.group(5)+line+sep)
                    self.offset += len(line+sep)+1
                    continue
                elif m.group(5): # Old escape syntax
                    depr('Escape code lines with a backslash.') #0.12
                    line, sep, _ = self.source[self.offset:].partition('\n')
                    self.text_buffer.append(m.group(2)+line+sep)
                    self.offset += len(line+sep)+1
                    continue
                self.flush_text()
                self.read_code(multiline=bool(m.group(4)))
            else: break
        self.text_buffer.append(self.source[self.offset:])
        self.flush_text()
        return ''.join(self.code_buffer)

    def read_code(self, multiline):
        code_line, comment = '', ''
        while True:
            m = self.re_tok.search(self.source[self.offset:])
            if not m:
                code_line += self.source[self.offset:]
                self.offset = len(self.source)
                self.write_code(code_line.strip(), comment)
                return
            code_line += self.source[self.offset:self.offset+m.start()]
            self.offset += m.end()
            _str, _com, _po, _pc, _blk1, _blk2, _end, _cend, _nl = m.groups()
            if (code_line or self.paren_depth > 0) and (_blk1 or _blk2): # a if b else c
                code_line += _blk1 or _blk2
                continue
            if _str:    # Python string
                code_line += _str
            elif _com:  # Python comment (up to EOL)
                comment = _com
                if multiline and _com.strip().endswith(self._tokens[1]):
                    multiline = False # Allow end-of-block in comments
            elif _po:  # open parenthesis
                self.paren_depth += 1
                code_line += _po
            elif _pc:  # close parenthesis
                if self.paren_depth > 0:
                    # we could check for matching parentheses here, but it's
                    # easier to leave that to python - just check counts
                    self.paren_depth -= 1
                code_line += _pc
            elif _blk1: # Start-block keyword (if/for/while/def/try/...)
                code_line, self.indent_mod = _blk1, -1
                self.indent += 1
            elif _blk2: # Continue-block keyword (else/elif/except/...)
                code_line, self.indent_mod = _blk2, -1
            elif _end:  # The non-standard 'end'-keyword (ends a block)
                self.indent -= 1
            elif _cend: # The end-code-block template token (usually '%>')
                if multiline: multiline = False
                else: code_line += _cend
            else: # \n
                self.write_code(code_line.strip(), comment)
                self.lineno += 1
                code_line, comment, self.indent_mod = '', '', 0
                if not multiline:
                    break

    def flush_text(self):
        text = ''.join(self.text_buffer)
        del self.text_buffer[:]
        if not text: return
        parts, pos, nl = [], 0, '\\\n'+'  '*self.indent
        for m in self.re_inl.finditer(text):
            prefix, pos = text[pos:m.start()], m.end()
            if prefix:
                parts.append(nl.join(map(repr, prefix.splitlines(True))))
            if prefix.endswith('\n'): parts[-1] += nl
            parts.append(self.process_inline(m.group(1).strip()))
        if pos < len(text):
            prefix = text[pos:]
            lines = prefix.splitlines(True)
            if lines[-1].endswith('\\\\\n'): lines[-1] = lines[-1][:-3]
            elif lines[-1].endswith('\\\\\r\n'): lines[-1] = lines[-1][:-4]
            parts.append(nl.join(map(repr, lines)))
        code = '_printlist((%s,))' % ', '.join(parts)
        self.lineno += code.count('\n')+1
        self.write_code(code)

    def process_inline(self, chunk):
        if chunk[0] == '!': return '_str(%s)' % chunk[1:]
        return '_escape(%s)' % chunk

    def write_code(self, line, comment=''):
        line, comment = self.fix_backward_compatibility(line, comment)
        code  = '  ' * (self.indent+self.indent_mod)
        code += line.lstrip() + comment + '\n'
        self.code_buffer.append(code)

    def fix_backward_compatibility(self, line, comment):
        parts = line.strip().split(None, 2)
        if parts and parts[0] in ('include', 'rebase'):
            depr('The include and rebase keywords are functions now.') #0.12
            if len(parts) == 1:   return "_printlist([base])", comment
            elif len(parts) == 2: return "_=%s(%r)" % tuple(parts), comment
            else:                 return "_=%s(%r, %s)" % tuple(parts), comment
        if self.lineno <= 2 and not line.strip() and 'coding' in comment:
            m = re.match(r"#.*coding[:=]\s*([-\w.]+)", comment)
            if m:
                depr('PEP263 encoding strings in templates are deprecated.') #0.12
                enc = m.group(1)
                self.source = self.source.encode(self.encoding).decode(enc)
                self.encoding = enc
                return line, comment.replace('coding','coding*')
        return line, comment


def template(*args, **kwargs):
    '''
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    '''
    tpl = args[0] if args else None
    adapter = kwargs.pop('template_adapter', SimpleTemplate)
    lookup = kwargs.pop('template_lookup', TEMPLATE_PATH)
    tplid = (id(lookup), tpl)
    if tplid not in TEMPLATES or DEBUG:
        settings = kwargs.pop('template_settings', {})
        if isinstance(tpl, adapter):
            TEMPLATES[tplid] = tpl
            if settings: TEMPLATES[tplid].prepare(**settings)
        elif "\n" in tpl or "{" in tpl or "%" in tpl or '$' in tpl:
            TEMPLATES[tplid] = adapter(source=tpl, lookup=lookup, **settings)
        else:
            TEMPLATES[tplid] = adapter(name=tpl, lookup=lookup, **settings)
    if not TEMPLATES[tplid]:
        abort(500, 'Template (%s) not found' % tpl)
    for dictarg in args[1:]: kwargs.update(dictarg)
    return TEMPLATES[tplid].render(kwargs)

mako_template = functools.partial(template, template_adapter=MakoTemplate)
cheetah_template = functools.partial(template, template_adapter=CheetahTemplate)
jinja2_template = functools.partial(template, template_adapter=Jinja2Template)


def view(tpl_name, **defaults):
    ''' Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, (dict, DictMixin)):
                tplvars = defaults.copy()
                tplvars.update(result)
                return template(tpl_name, **tplvars)
            elif result is None:
                return template(tpl_name, defaults)
            return result
        return wrapper
    return decorator

mako_view = functools.partial(view, template_adapter=MakoTemplate)
cheetah_view = functools.partial(view, template_adapter=CheetahTemplate)
jinja2_view = functools.partial(view, template_adapter=Jinja2Template)






###############################################################################
# Constants and Globals ########################################################
###############################################################################


TEMPLATE_PATH = ['./', './views/', '../views/']
TEMPLATES = {}
DEBUG = False
NORUN = False # If set, run() does nothing. Used by load_app()

#: A dict to map HTTP status codes (e.g. 404) to phrases (e.g. 'Not Found')
HTTP_CODES = httplib.responses
HTTP_CODES[418] = "I'm a teapot" # RFC 2324
HTTP_CODES[422] = "Unprocessable Entity" # RFC 4918
HTTP_CODES[428] = "Precondition Required"
HTTP_CODES[429] = "Too Many Requests"
HTTP_CODES[431] = "Request Header Fields Too Large"
HTTP_CODES[511] = "Network Authentication Required"
_HTTP_STATUS_LINES = dict((k, '%d %s'%(k,v)) for (k,v) in HTTP_CODES.items())

#: The default template used for error pages. Override with @error()
ERROR_PAGE_TEMPLATE = """
%%try:
    %%from %s import DEBUG, HTTP_CODES, request, touni
    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
    <html>
        <head>
            <title>Error: {{e.status}}</title>
            <style type="text/css">
              html {background-color: #eee; font-family: sans;}
              body {background-color: #fff; border: 1px solid #ddd;
                    padding: 15px; margin: 15px;}
              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}
            </style>
        </head>
        <body>
            <h1>Error: {{e.status}}</h1>
            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>
               caused an error:</p>
            <pre>{{e.body}}</pre>
            %%if DEBUG and e.exception:
              <h2>Exception:</h2>
              <pre>{{repr(e.exception)}}</pre>
            %%end
            %%if DEBUG and e.traceback:
              <h2>Traceback:</h2>
              <pre>{{e.traceback}}</pre>
            %%end
        </body>
    </html>
%%except ImportError:
    <b>ImportError:</b> Could not generate the error page. Please add bottle to
    the import path.
%%end
""" % __name__

#: A thread-safe instance of :class:`LocalRequest`. If accessed from within a
#: request callback, this instance always refers to the *current* request
#: (even on a multithreaded server).
request = LocalRequest()

#: A thread-safe instance of :class:`LocalResponse`. It is used to change the
#: HTTP response for the *current* request.
response = LocalResponse()

#: A thread-safe namespace. Not used by Bottle.
local = threading.local()

# Initialize app stack (create first empty Bottle app)
# BC: 0.6.4 and needed for run()
app = default_app = AppStack()
app.push()

#: A virtual package that redirects import statements.
#: Example: ``import bottle.ext.sqlite`` actually imports `bottle_sqlite`.
ext = _ImportRedirect('bottle.ext' if __name__ == '__main__' else __name__+".ext", 'bottle_%s').module

if __name__ == '__main__':
    opt, args, parser = _cmd_options, _cmd_args, _cmd_parser
    if opt.version:
        _stdout('Bottle %s\n'%__version__)
        sys.exit(0)
    if not args:
        parser.print_help()
        _stderr('\nError: No application specified.\n')
        sys.exit(1)

    sys.path.insert(0, '.')
    sys.modules.setdefault('bottle', sys.modules['__main__'])

    host, port = (opt.bind or 'localhost'), 8080
    if ':' in host and host.rfind(']') < host.rfind(':'):
        host, port = host.rsplit(':', 1)
    host = host.strip('[]')

    run(args[0], host=host, port=int(port), server=opt.server,
        reloader=opt.reload, plugins=opt.plugin, debug=opt.debug)




# THE END
Û
∑∆Xc           @Ä  sá  d  Z  d d l m Z d Z d Z d Z e d k r5d d l m Z e d d	 É Z	 e	 j
 Z e d
 d d d d Ée d d d d d d Ée d d d d d d Ée d d d d d d Ée d d d d d Ée d d d d d  Ée	 j É  \ Z Z e j oe j j d! É r2d d" l Z e j j É  n  n  d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l  Z  d d" l! Z! d d" l" Z" d d" l# Z# d d" l$ Z$ d d" l% Z% d d# l& m' Z( m& Z& m) Z) d d$ l" m* Z* d d% l+ m, Z, m- Z- d d& l. m/ Z/ d d' l0 m1 Z1 y d d( l2 m3 Z4 m5 Z6 Wn| e7 k
 rÔy d d( l8 m3 Z4 m5 Z6 WnN e7 k
 rÎy d d( l9 m3 Z4 m5 Z6 Wn  e7 k
 rÁd) Ñ  Z4 e4 Z6 n Xn Xn Xe! j: Z; e; d* d+ d+ f k Z< e; d, d- d+ f k  Z= d* d. d+ f e; k oLd* d, d+ f k  n Z> d/ Ñ  Z? y" e! j@ jA e! jB jA f \ ZC ZD Wn# eE k
 r°d0 Ñ  ZC d1 Ñ  ZD n Xe< rÜd d" lF jG ZH d d" lI ZJ d d2 lK mL ZL mM ZN d d3 lK mO ZO mP ZQ mR ZS e jT eS d4 d5 ÉZS d d6 lU mV ZV d d7 lW mX ZY d d" lZ ZZ d d8 l[ m\ Z\ d d9 l] m^ Z^ e_ Z` e_ Za d: Ñ  Zb d; Ñ  Zc ed Ze d< Ñ  Zf nd d" lH ZH d d" lJ ZJ d d2 lg mL ZL mM ZN d d3 lh mO ZO mP ZQ mR ZS d d6 li mV ZV d d= l me Ze d d" lj ZZ d d> lk mk Z\ d d? l^ ml Z^ e= rZd@ Zm e% jn em eo É d dA lp mY ZY dB Ñ  Zq e_ Zr n d d7 lW mX ZY ea Za e6 Zb es et dC dD dE É É dF dG Ñ Zu dF dH dI Ñ Zv e< r≥ev n eu Zw e> rËd dJ l[ mx Zx dK ex f dL Ñ  É  YZy n  dM Ñ  Zz e{ dN Ñ Z| dO Ñ  Z} dP e~ f dQ Ñ  É  YZ dR e~ f dS Ñ  É  YZÄ dT e~ f dU Ñ  É  YZÅ dV eÇ f dW Ñ  É  YZÉ dX eÉ f dY Ñ  É  YZÑ dZ eÉ f d[ Ñ  É  YZÖ d\ eÑ f d] Ñ  É  YZÜ d^ eÑ f d_ Ñ  É  YZá d` eÑ f da Ñ  É  YZà db Ñ  Zâ dc e~ f dd Ñ  É  YZä de e~ f df Ñ  É  YZã dg e~ f dh Ñ  É  YZå di e~ f dj Ñ  É  YZç dk Ñ  Zé dl e~ f dm Ñ  É  YZè dn e~ f do Ñ  É  YZê eë dp Ñ Zí dq eç f dr Ñ  É  YZì ds eê f dt Ñ  É  YZî eç Zï eê Zñ du eñ eÉ f dv Ñ  É  YZó dw eó f dx Ñ  É  YZò dy eÉ f dz Ñ  É  YZô d{ e~ f d| Ñ  É  YZö d} e~ f d~ Ñ  É  YZõ d e~ f dÄ Ñ  É  YZú dÅ eY f dÇ Ñ  É  YZù dÉ eù f dÑ Ñ  É  YZû dÖ eù f dÜ Ñ  É  YZü dá eY f dà Ñ  É  YZ† dâ e° f dä Ñ  É  YZ¢ dã e£ f då Ñ  É  YZ§ dç e~ f dé Ñ  É  YZ• dè e~ f dê Ñ  É  YZ¶ dë e~ f dí Ñ  É  YZß dì e~ f dî Ñ  É  YZ® dï dñ dó Ñ Z© eë dò Ñ Z™ dô dô dö Ñ Z´ dõ e{ dú dù Ñ Z¨ e≠ dû Ñ ZÆ dü Ñ  ZØ d† Ñ  Z∞ d° Ñ  Z± d+ d¢ Ñ Z≤ d£ Ñ  Z≥ d§ Ñ  Z¥ d• Ñ  Zµ d¶ Ñ  Z∂ dß Ñ  Z∑ d® Ñ  Z∏ d© Ñ  Zπ d™ Ñ  Z∫ d. d´ Ñ Zª d¨ d≠ dÆ Ñ Zº dØ Ñ  ZΩ eΩ d∞ É Zæ eΩ d± É Zø eΩ d≤ É Z¿ eΩ d≥ É Z¡ eΩ d¥ É Z¬ eΩ dµ É Z√ eΩ d∂ É Zƒ eΩ d∑ É Z≈ eΩ d∏ É Z∆ eΩ dπ É Z« eΩ d∫ É Z» dª e~ f dº Ñ  É  YZ… dΩ e… f dæ Ñ  É  YZ  dø e… f d¿ Ñ  É  YZÀ d¡ e… f d¬ Ñ  É  YZÃ d√ e… f dƒ Ñ  É  YZÕ d≈ e… f d∆ Ñ  É  YZŒ d« e… f d» Ñ  É  YZœ d… e… f d  Ñ  É  YZ– dÀ e… f dÃ Ñ  É  YZ— dÕ e… f dŒ Ñ  É  YZ“ dœ e… f d– Ñ  É  YZ” d— e… f d“ Ñ  É  YZ‘ d” e… f d‘ Ñ  É  YZ’ d’ e… f d÷ Ñ  É  YZ÷ d◊ e… f dÿ Ñ  É  YZ◊ dŸ e… f d⁄ Ñ  É  YZÿ d€ e… f d‹ Ñ  É  YZŸ d› e… f dﬁ Ñ  É  YZ⁄ dﬂ e… f d‡ Ñ  É  YZ€ d· e… f d‚ Ñ  É  YZ‹ i e  d„ 6eÀ d‰ 6eÃ d 6eŒ dÂ 6eÕ dÊ 6eœ dÁ 6e— dË 6e“ dÈ 6e” dÍ 6e‘ dÎ 6e’ dÏ 6e– dÌ 6eÿ dÓ 6eŸ dÔ 6e÷ d! 6e◊ d 6e⁄ dÒ 6e€ dÚ 6e‹ dõ 6Z› dÛ Ñ  Zﬁ dÙ Ñ  Zﬂ eÆ Z‡ eë d dı dˆ d. e{ e{ eë eë d˜ Ñ	 Z· d¯ e# j‚ f d˘ Ñ  É  YZ„ d˙ eò f d˚ Ñ  É  YZ‰ d¸ e~ f d˝ Ñ  É  YZÂ d˛ eÂ f dˇ Ñ  É  YZÊ d eÂ f dÑ  É  YZÁ deÂ f dÑ  É  YZË deÂ f dÑ  É  YZÈ de‰ f dÑ  É  YZÍ de~ f d	Ñ  É  YZÎ d
Ñ  ZÏ e jT eÏ deÊ ÉZÌ e jT eÏ deÁ ÉZÓ e jT eÏ deË ÉZÔ dÑ  Z e jT e deÊ ÉZÒ e jT e deÁ ÉZÚ e jT e deË ÉZÛ dddg ZÙ i  Zı e{ aˆ e{ a˜ eH j¯ Z˘ de˘ d<de˘ d<de˘ d<de˘ d<de˘ d<de˘ d<e° dÑ  e˘ j˙ É  DÉ É Z˚ de Z¸ eì É  Z˝ eî É  Z˛ e# jˇ É  Zˇ e§ É  Z Ze jÉ  eú e d k rdn e dd É jZe d k rÉe e e	 f \ ZZZejrueC d!e É e! j	d+ É n  er†ej
É  eD d"É e! j	d. É n  e! jjd+ d#É e! jjd$e! jd É ejpŸd%dˆ f \ ZZd&ek oejd'É ejd&É k  r-ejd&d. É \ ZZn  ejd(É Ze· ed+ d)ed*eeÉ d+ej d,ejd-ejd.ejÆ Én  d" S(/  sÕ  
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
iˇˇˇˇ(   t   with_statements   Marcel Hellkamps   0.12.9t   MITt   __main__(   t   OptionParsert   usages)   usage: %prog [options] package.module:apps	   --versiont   actiont
   store_truet   helps   show version number.s   -bs   --bindt   metavart   ADDRESSs   bind socket to ADDRESS.s   -ss   --servert   defaultt   wsgirefs   use SERVER as backend.s   -ps   --plugint   appends   install additional plugin/s.s   --debugs   start server in debug mode.s   --reloads   auto-reload on file changes.t   geventN(   t   datet   datetimet	   timedelta(   t   TemporaryFile(   t
   format_exct	   print_exc(   t
   getargspec(   t	   normalize(   t   dumpst   loadsc         CÄ  s   t  d É Ç d  S(   Ns/   JSON support requires Python 2.6 or simplejson.(   t   ImportError(   t   data(    (    s&   /home/lgardner/git/professor/bottle.pyt
   json_dumps6   s    i   i    i   i   i   c           CÄ  s   t  j É  d S(   Ni   (   t   syst   exc_info(    (    (    s&   /home/lgardner/git/professor/bottle.pyt   _eE   s    c         CÄ  s   t  j j |  É S(   N(   R   t   stdoutt   write(   t   x(    (    s&   /home/lgardner/git/professor/bottle.pyt   <lambda>L   s    c         CÄ  s   t  j j |  É S(   N(   R   t   stderrR   (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   M   s    (   t   urljoint   SplitResult(   t	   urlencodet   quotet   unquotet   encodingt   latin1(   t   SimpleCookie(   t   MutableMapping(   t   BytesIO(   t   ConfigParserc         CÄ  s   t  t |  É É S(   N(   t   json_ldst   touni(   t   s(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]   s    c         CÄ  s   t  |  d É S(   Nt   __call__(   t   hasattr(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ^   s    c          GÄ  s%   |  d |  d É j  |  d É Ç d  S(   Ni    i   i   (   t   with_traceback(   t   a(    (    s&   /home/lgardner/git/professor/bottle.pyt   _raise`   s    (   t   imap(   t   StringIO(   t   SafeConfigParsers?   Python 2.5 support may be dropped in future versions of Bottle.(   t	   DictMixinc         CÄ  s
   |  j  É  S(   N(   t   next(   t   it(    (    s&   /home/lgardner/git/professor/bottle.pyR:   o   s    s&   def _raise(*a): raise a[0], a[1], a[2]s   <py3fix>t   exect   utf8c         CÄ  s&   t  |  t É r |  j | É St |  É S(   N(   t
   isinstancet   unicodet   encodet   bytes(   R0   t   enc(    (    s&   /home/lgardner/git/professor/bottle.pyt   tobx   s    t   strictc         CÄ  s)   t  |  t É r |  j | | É St |  É S(   N(   R>   RA   t   decodeR?   (   R0   RB   t   err(    (    s&   /home/lgardner/git/professor/bottle.pyR/   z   s    (   t   TextIOWrappert   NCTextIOWrapperc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s   d  S(   N(    (   t   self(    (    s&   /home/lgardner/git/professor/bottle.pyt   closeÉ   s    (   t   __name__t
   __module__RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRH   Ç   s   c         OÄ  s2   y t  j |  | | | é Wn t k
 r- n Xd  S(   N(   t	   functoolst   update_wrappert   AttributeError(   t   wrappert   wrappedR4   t   ka(    (    s&   /home/lgardner/git/professor/bottle.pyRN   á   s      c         CÄ  s   t  j |  t d d Éd  S(   Nt
   stackleveli   (   t   warningst   warnt   DeprecationWarning(   t   messaget   hard(    (    s&   /home/lgardner/git/professor/bottle.pyt   deprê   s    c         CÄ  s:   t  |  t t t t f É r% t |  É S|  r2 |  g Sg  Sd  S(   N(   R>   t   tuplet   listt   sett   dict(   R   (    (    s&   /home/lgardner/git/professor/bottle.pyt   makelistì   s
     
 t   DictPropertyc           BÄ  sA   e  Z d  Z d e d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 RS(   s=    Property that maps to a key in a local dict-like attribute. c         CÄ  s!   | | | |  _  |  _ |  _ d  S(   N(   t   attrt   keyt	   read_only(   RI   R`   Ra   Rb   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __init__õ   s    c         CÄ  s9   t  j |  | d g  É| |  j p( | j |  _ |  _ |  S(   Nt   updated(   RM   RN   Ra   RK   t   getter(   RI   t   func(    (    s&   /home/lgardner/git/professor/bottle.pyR1   û   s    c         CÄ  sV   | d  k r |  S|  j t | |  j É } } | | k rN |  j | É | | <n  | | S(   N(   t   NoneRa   t   getattrR`   Re   (   RI   t   objt   clsRa   t   storage(    (    s&   /home/lgardner/git/professor/bottle.pyt   __get__£   s      c         CÄ  s5   |  j  r t d É Ç n  | t | |  j É |  j <d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   t   value(    (    s&   /home/lgardner/git/professor/bottle.pyt   __set__©   s    	 c         CÄ  s2   |  j  r t d É Ç n  t | |  j É |  j =d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   __delete__≠   s    	 N(
   RK   RL   t   __doc__Rg   t   FalseRc   R1   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR_   ô   s   			t   cached_propertyc           BÄ  s    e  Z d  Z d Ñ  Z d Ñ  Z RS(   s•    A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. c         CÄ  s   t  | d É |  _ | |  _ d  S(   NRp   (   Rh   Rp   Rf   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ∑   s    c         CÄ  s4   | d  k r |  S|  j | É } | j |  j j <| S(   N(   Rg   Rf   t   __dict__RK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   ª   s      (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRr   ≤   s   	t   lazy_attributec           BÄ  s    e  Z d  Z d Ñ  Z d Ñ  Z RS(   s4    A property that caches itself to the class object. c         CÄ  s#   t  j |  | d g  É| |  _ d  S(   NRd   (   RM   RN   Re   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   √   s    c         CÄ  s&   |  j  | É } t | |  j | É | S(   N(   Re   t   setattrRK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   «   s    (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt   ¡   s   	t   BottleExceptionc           BÄ  s   e  Z d  Z RS(   s-    A base class for exceptions used by bottle. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRv   ÷   s   t
   RouteErrorc           BÄ  s   e  Z d  Z RS(   s9    This is a base class for all routing related exceptions (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw   ‰   s   t
   RouteResetc           BÄ  s   e  Z d  Z RS(   sf    If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx   Ë   s   t   RouterUnknownModeErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRy   Ï   s    t   RouteSyntaxErrorc           BÄ  s   e  Z d  Z RS(   s@    The route parser found something not supported by this router. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRz   Ô   s   t   RouteBuildErrorc           BÄ  s   e  Z d  Z RS(   s    The route could not be built. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{   Û   s   c         CÄ  s&   d |  k r |  St  j d d Ñ  |  É S(   s^    Turn all capturing groups in a regular expression pattern into
        non-capturing groups. t   (s   (\\*)(\(\?P<[^>]+>|\((?!\?))c         SÄ  s7   t  |  j d É É d r& |  j d É S|  j d É d S(   Ni   i   i    s   (?:(   t   lent   group(   t   m(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ¸   s    (   t   ret   sub(   t   p(    (    s&   /home/lgardner/git/professor/bottle.pyt   _re_flatten˜   s     	t   Routerc           BÄ  st   e  Z d  Z d Z d Z d Z e d Ñ Z d Ñ  Z e	 j
 d É Z d Ñ  Z d d Ñ Z d	 Ñ  Z d
 Ñ  Z d Ñ  Z RS(   sA   A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    s   [^/]+RÄ   ic   c         Ä  sz   g  à  _  i  à  _ i  à  _ i  à  _ i  à  _ i  à  _ | à  _ i á  f d Ü  d 6d Ñ  d 6d Ñ  d 6d Ñ  d 6à  _ d  S(	   Nc         Ä  s   t  |  p à  j É d  d  f S(   N(   RÉ   t   default_patternRg   (   t   conf(   RI   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    RÄ   c         SÄ  s   d t  d Ñ  f S(   Ns   -?\d+c         SÄ  s   t  t |  É É S(   N(   t   strt   int(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   Rà   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    Rà   c         SÄ  s   d t  d Ñ  f S(   Ns   -?[\d.]+c         SÄ  s   t  t |  É É S(   N(   Rá   t   float(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   Râ   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    Râ   c         SÄ  s   d S(   Ns   .+?(   s   .+?NN(   Rg   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!      s    t   path(   t   rulest   _groupst   buildert   statict   dyna_routest   dyna_regexest   strict_ordert   filters(   RI   RD   (    (   RI   s&   /home/lgardner/git/professor/bottle.pyRc     s    							

c         CÄ  s   | |  j  | <d S(   s‚    Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. N(   Rí   (   RI   t   nameRf   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   add_filter"  s    sÄ   (\\*)(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)(?::((?:\\.|[^\\>]+)+)?)?)?>))c   	      cÄ  s?  d \ } } x˚ |  j  j | É D]Á } | | | | j É  !7} | j É  } t | d É d rè | | j d É t | d É 7} | j É  } q n  | r¶ | d  d  f Vn  | d d  k r√ | d d !n
 | d d !\ } } } | | pÂ d | pÓ d  f V| j É  d } } q W| t | É k s"| r;| | | d  d  f Vn  d  S(	   Ni    t    i   i   i   i   R
   (   i    Rï   (   t   rule_syntaxt   finditert   startt   groupsR}   R~   t   endRg   (	   RI   t   rulet   offsett   prefixt   matcht   gRì   t   filtrRÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _itertokens-  s    !3c         Ä  s˛  d } g  } d } g  â  g  } t  }	 x|  j | É D]\ }
 } } | rt }	 | d k rg |  j } n  |  j | | É \ } } } |
 sÆ | d | 7} d | }
 | d 7} n! | d |
 | f 7} | j |
 É | rÎ à  j |
 | f É n  | j |
 | p˝ t f É q4 |
 r4 | t j |
 É 7} | j d |
 f É q4 q4 W| |  j
 | <| r]| |  j
 | <n  |	 r§|  j r§|  j j | i  É | d f |  j | |  j | É <d Sy  t j d	 | É } | j â Wn- t j k
 rÛt d
 | t É  f É Ç n Xà  rá  á f d Ü  } n! | j r*á f d Ü  } n d } t | É } | | | | f } | | f |  j k r≠t råd } t j | | | f t É n  | |  j | |  j | | f <n@ |  j j | g  É j | É t |  j | É d |  j | | f <|  j | É d S(   s<    Add a new rule or replace the target for an existing rule. i    Rï   R
   s   (?:%s)s   anon%di   s
   (?P<%s>%s)Ns   ^(%s)$s   Could not add Route: %s (%s)c         Ä  sh   à |  É j  É  } xO à  D]G \ } } y | | | É | | <Wq t k
 r_ t d d É Ç q Xq W| S(   Niê  s   Path has wrong format.(   t	   groupdictt
   ValueErrort	   HTTPError(   Rä   t   url_argsRì   t   wildcard_filter(   Rí   t   re_match(    s&   /home/lgardner/git/professor/bottle.pyt   getargsh  s    c         Ä  s   à  |  É j  É  S(   N(   R¢   (   Rä   (   Rß   (    s&   /home/lgardner/git/professor/bottle.pyR®   q  s    s3   Route <%s %s> overwrites a previously defined route(   t   TrueR°   Rq   t   default_filterRí   R   Rá   RÄ   t   escapeRg   Rç   Rë   Ré   t
   setdefaultt   buildt   compileRû   t   errorRz   R   t
   groupindexRÉ   Rå   t   DEBUGRT   RU   t   RuntimeWarningRè   R}   t   _compile(   RI   Rõ   t   methodt   targetRì   t   anonst   keyst   patternRç   t	   is_staticRa   t   modeRÜ   t   maskt	   in_filtert
   out_filtert
   re_patternR®   t   flatpatt
   whole_rulet   msg(    (   Rí   Rß   s&   /home/lgardner/git/professor/bottle.pyt   add>  sf     
   	!$c         CÄ  sÿ   |  j  | } g  } |  j | <|  j } x™ t d t | É | É D]ê } | | | | !} d Ñ  | DÉ } d j d Ñ  | DÉ É } t j | É j } g  | D] \ } } }	 }
 |	 |
 f ^ qô } | j	 | | f É q@ Wd  S(   Ni    c         sÄ  s!   |  ] \ } } } } | Vq d  S(   N(    (   t   .0t   _Rø   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>ä  s    t   |c         sÄ  s   |  ] } d  | Vq d S(   s   (^%s$)N(    (   R√   Rø   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>ã  s    (
   Rè   Rê   t   _MAX_GROUPS_PER_PATTERNt   rangeR}   t   joinRÄ   RÆ   Rû   R   (   RI   R¥   t	   all_rulest
   comborulest	   maxgroupsR    t   somet   combinedRƒ   Rµ   R®   Rã   (    (    s&   /home/lgardner/git/professor/bottle.pyR≥   Ñ  s    	+c   
      OÄ  sÍ   |  j  j | É } | s* t d | É Ç n  yé x( t | É D] \ } } | | d | <q: Wd j g  | D]- \ } } | rå | | j | É É n | ^ qe É }	 | s• |	 S|	 d t | É SWn+ t k
 rÂ t d t É  j	 d É Ç n Xd S(   s2    Build an URL by filling the wildcards in a rule. s   No route with that name.s   anon%dRï   t   ?s   Missing URL argument: %ri    N(
   Rç   t   getR{   t	   enumerateR»   t   popR%   t   KeyErrorR   t   args(
   RI   t   _nameR∂   t   queryRç   t   iRm   t   nt   ft   url(    (    s&   /home/lgardner/git/professor/bottle.pyR≠   ê  s      C c         CÄ  s<  | d j  É  } | d p d } d } | d k rG d | d d g } n d | d g } xÿ | D]– } | |  j k r∏ | |  j | k r∏ |  j | | \ } } | | r± | | É n i  f S| |  j k r] xc |  j | D]Q \ } }	 | | É }
 |
 r’ |	 |
 j d \ } } | | r| | É n i  f Sq’ Wq] q] Wt g  É } t | É } x> t |  j É | D]) } | |  j | k r]| j | É q]q]Wx_ t |  j É | | D]F } x= |  j | D]. \ } }	 | | É }
 |
 r∂| j | É q∂q∂Wq¢W| rd	 j t | É É } t	 d
 d d | ÉÇ n  t	 d d t
 | É É Ç d S(   sD    Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). t   REQUEST_METHODt	   PATH_INFOt   /t   HEADt   PROXYt   GETt   ANYi   t   ,iï  s   Method not allowed.t   Allowiî  s   Not found: N(   t   upperRg   Ré   Rê   t	   lastindexR\   R¬   R»   t   sortedR§   t   repr(   RI   t   environt   verbRä   Rµ   t   methodsR¥   R®   RÕ   Rã   Rû   t   allowedt   nocheckt   allow_header(    (    s&   /home/lgardner/git/professor/bottle.pyRû   õ  s<    "'N(   RK   RL   Rp   RÖ   R™   R∆   Rq   Rc   Rî   RÄ   RÆ   Rñ   R°   Rg   R¬   R≥   R≠   Rû   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÑ   ˇ   s   
		F		t   Routec           BÄ  sí   e  Z d  Z d d d d Ñ Z d Ñ  Z e d Ñ  É Z d Ñ  Z d Ñ  Z	 e
 d Ñ  É Z d Ñ  Z d Ñ  Z d	 Ñ  Z d
 Ñ  Z d d Ñ Z d Ñ  Z RS(   sÓ    This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    c   	      KÄ  sp   | |  _  | |  _ | |  _ | |  _ | p- d  |  _ | p< g  |  _ | pK g  |  _ t É  j	 | d t
 É|  _ d  S(   Nt   make_namespaces(   t   appRõ   R¥   t   callbackRg   Rì   t   pluginst   skiplistt
   ConfigDictt	   load_dictR©   t   config(	   RI   RÔ   Rõ   R¥   R   Rì   RÒ   RÚ   Rı   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Õ  s    				c         OÄ  s   t  d É |  j | | é  S(   Nsî   Some APIs changed to return Route() instances instead of callables. Make sure to use the Route.call method and not to call Route instances directly.(   RY   t   call(   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ‚  s    
c         CÄ  s
   |  j  É  S(   sç    The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.(   t   _make_callback(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRˆ   Ë  s    c         CÄ  s   |  j  j d d É d S(   sk    Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. Rˆ   N(   Rs   R—   Rg   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   resetÓ  s    c         CÄ  s   |  j  d S(   s:    Do all on-demand work immediately (useful for debugging).N(   Rˆ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   prepareÛ  s    c         CÄ  sY   t  d É t d |  j d |  j d |  j d |  j d |  j d |  j d |  j d	 |  j	 É S(
   Ns=   Switch to Plugin API v2 and access the Route object directly.Rõ   R¥   R   Rì   RÔ   Rı   t   applyt   skip(
   RY   R]   Rõ   R¥   R   Rì   RÔ   Rı   RÒ   RÚ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _context˜  s    
!c         cÄ  s¬   t  É  } x≤ t |  j j |  j É D]ó } t |  j k r< Pn  t | d t É } | ru | |  j k s# | | k ru q# n  | |  j k s# t | É |  j k rü q# n  | rµ | j	 | É n  | Vq# Wd S(   s)    Yield all Plugins affecting this route. Rì   N(
   R\   t   reversedRÔ   RÒ   R©   RÚ   Rh   Rq   t   typeR¬   (   RI   t   uniqueRÇ   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   all_plugins˛  s    	  ! $  c         CÄ  s¬   |  j  } x≤ |  j É  D]§ } ya t | d É rp t | d d É } | d k rR |  n |  j } | j | | É } n | | É } Wn t k
 ró |  j É  SX| |  j  k	 r t | |  j  É q q W| S(   NR˙   t   apii   (	   R   R   R2   Rh   R¸   R˙   Rx   R˜   RN   (   RI   R   t   pluginR  t   context(    (    s&   /home/lgardner/git/professor/bottle.pyR˜   	  s    	c         CÄ  sx   |  j  } t | t r d n d | É } t r3 d n d } x8 t | | É rs t | | É rs t | | É d j } q< W| S(   sq    Return the callback. If the callback is a decorated function, try to
            recover the original function. t   __func__t   im_funct   __closure__t   func_closurei    (   R   Rh   t   py3kR2   t   cell_contents(   RI   Rf   t   closure_attr(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_undecorated_callback  s    	!c         CÄ  s   t  |  j É  É d S(   s”    Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. i    (   R   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   get_callback_args#  s    c         CÄ  s8   x1 |  j  |  j j f D] } | | k r | | Sq W| S(   sp    Lookup a config field and return its value, first checking the
            route.config, then route.app.config.(   Rı   RÔ   t   conifg(   RI   Ra   R
   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_config)  s     c         CÄ  s#   |  j  É  } d |  j |  j | f S(   Ns
   <%s %r %r>(   R  R¥   Rõ   (   RI   t   cb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __repr__0  s    N(   RK   RL   Rp   Rg   Rc   R1   Rr   Rˆ   R¯   R˘   t   propertyR¸   R   R˜   R  R  R  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÌ   «  s   						
	t   Bottlec           BÄ  s[  e  Z d  Z e e d Ñ Z e d d É Z d& Z d Z e	 d Ñ  É Z
 d Ñ  Z d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d' d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d' d d' d' d' d' d Ñ Z d' d d Ñ Z d' d d Ñ Z d' d d Ñ Z d' d d Ñ Z d d  Ñ Z d! Ñ  Z  d" Ñ  Z! d' d# Ñ Z" d$ Ñ  Z# d% Ñ  Z$ RS((   s^   Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    c         CÄ  s‘   t  É  |  _ t j |  j d É |  j _ |  j j d d t É |  j j d d t É | |  j d <| |  j d <t É  |  _	 g  |  _
 t É  |  _ i  |  _ g  |  _ |  j d r¿ |  j t É  É n  |  j t É  É d  S(   NRı   t   autojsont   validatet   catchall(   RÛ   Rı   RM   t   partialt   trigger_hookt
   _on_changet   meta_sett   boolt   ResourceManagert	   resourcest   routesRÑ   t   routert   error_handlerRÒ   t   installt
   JSONPlugint   TemplatePlugin(   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   G  s    			Rı   R  t   before_requestt   after_requestt	   app_resetc         CÄ  s   t  d Ñ  |  j DÉ É S(   Nc         sÄ  s   |  ] } | g  f Vq d  S(   N(    (   R√   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>f  s    (   R]   t   _Bottle__hook_names(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hooksd  s    c         CÄ  sA   | |  j  k r) |  j | j d | É n |  j | j | É d S(   s´   Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        i    N(   t   _Bottle__hook_reversedR'  t   insertR   (   RI   Rì   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   add_hookh  s    c         CÄ  s>   | |  j  k r: | |  j  | k r: |  j  | j | É t Sd S(   s     Remove a callback from a hook. N(   R'  t   removeR©   (   RI   Rì   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   remove_hookx  s    "c         OÄ  s(   g  |  j  | D] } | | | é  ^ q S(   s.    Trigger a hook and return a list of results. (   R'  (   RI   t   _Bottle__nameR”   t   kwargst   hook(    (    s&   /home/lgardner/git/professor/bottle.pyR  ~  s    c         Ä  s   á  á f d Ü  } | S(   se    Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details.c         Ä  s   à j  à  |  É |  S(   N(   R*  (   Rf   (   Rì   RI   (    s&   /home/lgardner/git/professor/bottle.pyt	   decoratorÖ  s    (    (   RI   Rì   R0  (    (   Rì   RI   s&   /home/lgardner/git/professor/bottle.pyR/  Ç  s    c         Ä  s  t  à  t É r t d t É n  g  | j d É D] } | r/ | ^ q/ } | s\ t d É Ç n  t | É â á  á f d Ü  } | j d t É | j d d É | j d i | d	 6à  d
 6É | | d <|  j d d j	 | É | ç | j
 d É s|  j d d j	 | É | ç n  d S(   sø   Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        s*   Parameter order of Bottle.mount() changed.R‹   s   Empty path prefix.c          Ä  sî   z~ t  j à É t g  É â  d  á  f d Ü }  à t  j |  É } | rg à  j rg t j à  j | É } n  | ps à  j à  _ à  SWd  t  j à É Xd  S(   Nc         Ä  s[   | r! z t  | å  Wd  d  } Xn  |  à  _ x$ | D] \ } } à  j | | É q1 Wà  j j S(   N(   R5   Rg   t   statust
   add_headert   bodyR   (   R1  t
   headerlistR   Rì   Rm   (   t   rs(    s&   /home/lgardner/git/professor/bottle.pyt   start_response°  s    
	 (   t   requestt
   path_shiftt   HTTPResponseRg   RÁ   R3  t	   itertoolst   chain(   R6  R3  (   RÔ   t
   path_depth(   R5  s&   /home/lgardner/git/professor/bottle.pyt   mountpoint_wrapperù  s    	 R˚   R¥   Rﬁ   t
   mountpointRù   Rµ   R   s   /%s/<:re:.*>N(   R>   t
   basestringRY   R©   t   splitR£   R}   R¨   t   routeR»   t   endswith(   RI   Rù   RÔ   t   optionsRÇ   t   segmentsR=  (    (   RÔ   R<  s&   /home/lgardner/git/professor/bottle.pyt   mountä  s    ( 
c         CÄ  s=   t  | t É r | j } n  x | D] } |  j | É q" Wd S(   sÙ    Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. N(   R>   R  R  t	   add_route(   RI   R  RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   merge∫  s    c         CÄ  si   t  | d É r | j |  É n  t | É rK t  | d É rK t d É Ç n  |  j j | É |  j É  | S(   s‚    Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        t   setupR˙   s.   Plugins must be callable or implement .apply()(   R2   RH  t   callablet	   TypeErrorRÒ   R   R¯   (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR   ƒ  s     
c         CÄ  sœ   g  | } } x® t  t |  j É É d d d Ö D]Ñ \ } } | t k s~ | | k s~ | t | É k s~ t | d t É | k r0 | j | É |  j | =t | d É r¥ | j É  q¥ q0 q0 W| rÀ |  j	 É  n  | S(   s)   Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. NiˇˇˇˇRì   RJ   (
   R[   R–   RÒ   R©   R˛   Rh   R   R2   RJ   R¯   (   RI   R  t   removedR+  R÷   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   uninstall–  s    /*
  c         CÄ  sì   | d k r |  j } n+ t | t É r3 | g } n |  j | g } x | D] } | j É  qJ Wt rÇ x | D] } | j É  qk Wn  |  j d É d S(   s™    Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. R%  N(   Rg   R  R>   RÌ   R¯   R±   R˘   R  (   RI   RA  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR¯   ﬂ  s        c         CÄ  s=   x- |  j  D]" } t | d É r
 | j É  q
 q
 Wt |  _ d S(   s2    Close the application and all installed plugins. RJ   N(   RÒ   R2   RJ   R©   t   stopped(   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   Î  s     c         KÄ  s   t  |  | ç d S(   s-    Calls :func:`run` with the same parameters. N(   t   run(   RI   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ò  s    c         CÄ  s   |  j  j | É S(   s›    Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match.(   R  Rû   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRû   ı  s    c         KÄ  sV   t  j j d d É j d É d } |  j j | | ç j d É } t t d | É | É S(   s,    Return a string that matches a named route t   SCRIPT_NAMERï   R‹   (   R7  RÁ   Rœ   t   stripR  R≠   t   lstripR#   (   RI   t	   routenamet   kargst
   scriptnamet   location(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_url˚  s    "c         CÄ  sL   |  j  j | É |  j j | j | j | d | j Ét rH | j É  n  d S(   sS    Add a route object, but do not change the :data:`Route.app`
            attribute.Rì   N(	   R  R   R  R¬   Rõ   R¥   Rì   R±   R˘   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyRF    s    % Rﬂ   c   	      Ä  si   t  à É r d à â } n  t | É â t | É â á  á á á á á á f d Ü  } | re | | É S| S(   s   A decorator to bind a function to a request URL. Example::

                @app.route('/hello/:name')
                def hello(name):
                    return 'Hello %s' % name

            The ``:name`` part is a wildcard. See :class:`Router` for syntax
            details.

            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.

            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        c         Ä  sü   t  |  t É r t |  É }  n  xz t à É p6 t |  É D]` } xW t à É D]I } | j É  } t à | | |  d à d à d à à  ç} à j | É qJ Wq7 W|  S(   NRì   RÒ   RÚ   (   R>   R?  t   loadR^   t   yieldroutesR„   RÌ   RF  (   R   Rõ   RË   RA  (   Rı   R¥   Rì   Rä   RÒ   RI   RÚ   (    s&   /home/lgardner/git/professor/bottle.pyR0  &  s     N(   RI  Rg   R^   (	   RI   Rä   R¥   R   Rì   R˙   R˚   Rı   R0  (    (   Rı   R¥   Rì   Rä   RÒ   RI   RÚ   s&   /home/lgardner/git/professor/bottle.pyRA    s     !
c         KÄ  s   |  j  | | | ç S(   s    Equals :meth:`route`. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   2  s    t   POSTc         KÄ  s   |  j  | | | ç S(   s8    Equals :meth:`route` with a ``POST`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   post6  s    t   PUTc         KÄ  s   |  j  | | | ç S(   s7    Equals :meth:`route` with a ``PUT`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   put:  s    t   DELETEc         KÄ  s   |  j  | | | ç S(   s:    Equals :meth:`route` with a ``DELETE`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete>  s    iÙ  c         Ä  s   á  á f d Ü  } | S(   s<    Decorator: Register an output handler for a HTTP error codec         Ä  s   |  à j  t à  É <|  S(   N(   R  Rà   (   t   handler(   t   codeRI   (    s&   /home/lgardner/git/professor/bottle.pyRP   D  s    (    (   RI   R`  RP   (    (   R`  RI   s&   /home/lgardner/git/professor/bottle.pyRØ   B  s    c         CÄ  s   t  t t d | ÉÉ S(   Nt   e(   RC   t   templatet   ERROR_PAGE_TEMPLATE(   RI   t   res(    (    s&   /home/lgardner/git/professor/bottle.pyt   default_error_handlerI  s    c         CÄ  sã  | d } | d <t  rY y  | j d É j d É | d <WqY t k
 rU t d d É SXn  yä |  | d <t j | É t j É  zT |  j d É |  j	 j
 | É \ } } | | d	 <| | d
 <| | d <| j | ç  SWd  |  j d É XWn° t k
 r˙ t É  St k
 r| j É  |  j | É St t t f k
 r:Ç  nM t k
 rÜ|  j sVÇ  n  t É  } | d j | É t d d t É  | É SXd  S(   NR€   s   bottle.raw_pathR)   R=   iê  s#   Invalid path string. Expected UTF-8s
   bottle.appR#  s   route.handles   bottle.routes   route.url_argsR$  s   wsgi.errorsiÙ  s   Internal Server Error(   R  R@   RE   t   UnicodeErrorR§   R7  t   bindt   responseR  R  Rû   Rˆ   R9  R   Rx   R¯   t   _handlet   KeyboardInterruptt
   SystemExitt   MemoryErrort	   ExceptionR  R   R   (   RI   RÁ   Rä   RA  R”   t
   stacktrace(    (    s&   /home/lgardner/git/professor/bottle.pyRi  L  s>     





	 	c         CÄ  s$  | s# d t  k r d t  d <n  g  St | t t f É rn t | d t t f É rn | d d d !j | É } n  t | t É rí | j t  j É } n  t | t É r« d t  k r¿ t	 | É t  d <n  | g St | t
 É r| j t  É |  j j | j |  j É | É } |  j | É St | t É r=| j t  É |  j | j É St | d É ròd t j k rlt j d | É St | d É sãt | d É ròt | É Sn  y5 t | É } t | É } x | sÀt | É } q∂WWnä t k
 rÍ|  j d É St k
 rt É  } nW t t t f k
 rÇ  n; t k
 rY|  j s;Ç  n  t
 d d	 t É  t  É  É } n Xt | t É rv|  j | É St | t É rùt! j" | g | É } n_ t | t É r÷d
 Ñ  } t# | t! j" | g | É É } n& d t$ | É } |  j t
 d | É É St | d É r t% | | j& É } n  | S(   s˛    Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        s   Content-Lengthi    t   reads   wsgi.file_wrapperRJ   t   __iter__Rï   iÙ  s   Unhandled exceptionc         SÄ  s   |  j  t j É S(   N(   R@   Rh  t   charset(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   Æ  s    s   Unsupported response type: %s('   Rh  R>   RZ   R[   RA   R?   R»   R@   Rq  R}   R§   R˙   R  Rœ   t   status_codeRe  t   _castR9  R3  R2   R7  RÁ   t   WSGIFileWrappert   iterR:   t   StopIterationR   Rj  Rk  Rl  Rm  R  R   R:  R;  R6   R˛   t
   _closeiterRJ   (   RI   t   outt   peekt   ioutt   firstt   new_itert   encoderR¡   (    (    s&   /home/lgardner/git/professor/bottle.pyRs  o  sh    !		 	!c         CÄ  sE  yw |  j  |  j | É É } t j d k s: | d d k r_ t | d É rV | j É  n  g  } n  | t j t j É | SWn« t t	 t
 f k
 rñ Ç  n´ t k
 r@|  j s≤ Ç  n  d t | j d	 d
 É É } t r| d t t t É  É É t t É  É f 7} n  | d j | É d g } | d | t j É  É t | É g SXd S(   s    The bottle WSGI-interface. id   ie   iÃ   i0  R⁄   R›   RJ   s4   <h1>Critical error while processing request: %s</h1>R€   R‹   sD   <h2>Error:</h2>
<pre>
%s
</pre>
<h2>Traceback:</h2>
<pre>
%s
</pre>
s   wsgi.errorss   Content-Types   text/html; charset=UTF-8s   500 INTERNAL SERVER ERRORN(   id   ie   iÃ   i0  (   s   Content-Types   text/html; charset=UTF-8(   Rs  Ri  Rh  t   _status_codeR2   RJ   t   _status_lineR4  Rj  Rk  Rl  Rm  R  t   html_escapeRœ   R±   RÊ   R   R   R   R   R   RC   (   RI   RÁ   R6  Rx  RF   t   headers(    (    s&   /home/lgardner/git/professor/bottle.pyt   wsgi∑  s.     		 )	c         CÄ  s   |  j  | | É S(   s9    Each instance of :class:'Bottle' is a WSGI application. (   RÇ  (   RI   RÁ   R6  (    (    s&   /home/lgardner/git/professor/bottle.pyR1   —  s    (   s   before_requests   after_requests	   app_resets   configN(%   RK   RL   Rp   R©   Rc   R_   R  R&  R(  Rr   R'  R*  R,  R  R/  RE  RG  R   RL  Rg   R¯   RJ   RN  Rû   RV  RF  RA  Rœ   RZ  R\  R^  RØ   Re  Ri  Rs  RÇ  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  >  s@   					0	
							)		#H	t   BaseRequestc           BÄ  s;  e  Z d  Z d Z d Z d@ d Ñ Z e d d d e Éd Ñ  É Z	 e d d d e Éd Ñ  É Z
 e d d	 d e Éd
 Ñ  É Z e d Ñ  É Z e d Ñ  É Z e d d d e Éd Ñ  É Z d@ d Ñ Z e d d d e Éd Ñ  É Z d@ d@ d Ñ Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z d Ñ  Z d Ñ  Z e d d d e Éd  Ñ  É Z d! Ñ  Z e d" Ñ  É Z e d# Ñ  É Z e Z e d d$ d e Éd% Ñ  É Z e d& Ñ  É Z  e d d' d e Éd( Ñ  É Z! e d) Ñ  É Z" e d* Ñ  É Z# e d+ Ñ  É Z$ d, d- Ñ Z% e d. Ñ  É Z& e d/ Ñ  É Z' e d0 Ñ  É Z( e d1 Ñ  É Z) e d2 Ñ  É Z* e d3 Ñ  É Z+ e d4 Ñ  É Z, d5 Ñ  Z- d@ d6 Ñ Z. d7 Ñ  Z/ d8 Ñ  Z0 d9 Ñ  Z1 d: Ñ  Z2 d; Ñ  Z3 d< Ñ  Z4 d= Ñ  Z5 d> Ñ  Z6 d? Ñ  Z7 RS(A   sd   A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    RÁ   i ê c         CÄ  s,   | d k r i  n | |  _ |  |  j d <d S(   s!    Wrap a WSGI environ dictionary. s   bottle.requestN(   Rg   RÁ   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Ï  s    s
   bottle.appRb   c         CÄ  s   t  d É Ç d S(   s+    Bottle application handling this request. s0   This request is not connected to an application.N(   t   RuntimeError(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÔ   Û  s    s   bottle.routec         CÄ  s   t  d É Ç d S(   s=    The bottle :class:`Route` object that matches this request. s)   This request is not connected to a route.N(   RÑ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRA  ¯  s    s   route.url_argsc         CÄ  s   t  d É Ç d S(   s'    The arguments extracted from the URL. s)   This request is not connected to a route.N(   RÑ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR•   ˝  s    c         CÄ  s    d |  j  j d d É j d É S(   sÜ    The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). R‹   R€   Rï   (   RÁ   Rœ   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRä     s    c         CÄ  s   |  j  j d d É j É  S(   s6    The ``REQUEST_METHOD`` value as an uppercase string. R⁄   Rﬂ   (   RÁ   Rœ   R„   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¥     s    s   bottle.request.headersc         CÄ  s   t  |  j É S(   sf    A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. (   t   WSGIHeaderDictRÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÅ    s    c         CÄ  s   |  j  j | | É S(   sA    Return the value of a request header, or a given default value. (   RÅ  Rœ   (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_header  s    s   bottle.request.cookiesc         CÄ  s5   t  |  j j d d É É j É  } t d Ñ  | DÉ É S(   så    Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. t   HTTP_COOKIERï   c         sÄ  s!   |  ] } | j  | j f Vq d  S(   N(   Ra   Rm   (   R√   t   c(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R*   RÁ   Rœ   t   valuest	   FormsDict(   RI   t   cookies(    (    s&   /home/lgardner/git/professor/bottle.pyRã    s    !c         CÄ  sY   |  j  j | É } | rO | rO t | | É } | rK | d | k rK | d S| S| pX | S(   s   Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. i    i   (   Rã  Rœ   t   cookie_decode(   RI   Ra   R
   t   secretRm   t   dec(    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_cookie  s
    "s   bottle.request.queryc         CÄ  sT   t  É  } |  j d <t |  j j d d É É } x | D] \ } } | | | <q6 W| S(   s    The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. s
   bottle.gett   QUERY_STRINGRï   (   Rä  RÁ   t
   _parse_qslRœ   (   RI   Rœ   t   pairsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR’   )  s
    s   bottle.request.formsc         CÄ  sI   t  É  } x9 |  j j É  D]( \ } } t | t É s | | | <q q W| S(   s   Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. (   Rä  RY  t   allitemsR>   t
   FileUpload(   RI   t   formsRì   t   item(    (    s&   /home/lgardner/git/professor/bottle.pyRï  5  s
    	s   bottle.request.paramsc         CÄ  sa   t  É  } x' |  j j É  D] \ } } | | | <q Wx' |  j j É  D] \ } } | | | <qC W| S(   sâ    A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. (   Rä  R’   Rì  Rï  (   RI   t   paramsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRó  A  s    	s   bottle.request.filesc         CÄ  sI   t  É  } x9 |  j j É  D]( \ } } t | t É r | | | <q q W| S(   sò    File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        (   Rä  RY  Rì  R>   Rî  (   RI   t   filesRì   Rñ  (    (    s&   /home/lgardner/git/professor/bottle.pyRò  L  s
    	s   bottle.request.jsonc         CÄ  sX   |  j  j d d É j É  j d É d } | d k rT |  j É  } | sJ d St | É Sd S(   sÚ    If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. t   CONTENT_TYPERï   t   ;i    s   application/jsonN(   RÁ   Rœ   t   lowerR@  t   _get_body_stringRg   t
   json_loads(   RI   t   ctypet   b(    (    s&   /home/lgardner/git/professor/bottle.pyt   jsonX  s    (
c         cÄ  sW   t  d |  j É } x> | rR | t | | É É } | s: Pn  | V| t | É 8} q Wd  S(   Ni    (   t   maxt   content_lengtht   minR}   (   RI   Ro  t   bufsizet   maxreadt   part(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _iter_bodyf  s    	 c         cÄ  sï  t  d d É } t d É t d É t d É } } } xYt rê| d É } xT | d | k r† | d É } | | 7} | sÇ | Ç n  t | É | k rM | Ç qM qM W| j | É \ }	 }
 }
 y t t |	 j É  É d É } Wn t k
 rÒ | Ç n X| d	 k rPn  | } xg | d	 k rq| s5| t	 | | É É } n  | |  | | } } | sY| Ç n  | V| t | É 8} qW| d
 É | k r8 | Ç q8 q8 Wd  S(   Niê  s*   Error while parsing chunked transfer body.s   
Rö  Rï   i   i˛ˇˇˇi   i    i   (
   R§   RC   R©   R}   t	   partitionRà   t   tonatRP  R£   R£  (   RI   Ro  R§  RF   t   rnt   semt   bst   headerRà  t   sizeRƒ   R•  t   buffR¶  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _iter_chunkedn  s:    &	
 	 
  	s   bottle.request.bodyc         CÄ  sÂ   |  j  r |  j n |  j } |  j d j } t É  d t } } } xÇ | | |  j É D]n } | j | É | t	 | É 7} | rU | |  j k rU t
 d d É | } } | j | j É  É ~ t } qU qU W| |  j d <| j d É | S(   Ns
   wsgi.inputi    R∫   s   w+b(   t   chunkedR∞  Rß  RÁ   Ro  R,   Rq   t   MEMFILE_MAXR   R}   R   t   getvalueR©   t   seek(   RI   t	   body_itert	   read_funcR3  t	   body_sizet   is_temp_fileR¶  t   tmp(    (    s&   /home/lgardner/git/professor/bottle.pyt   _bodyâ  s    c         CÄ  sÉ   |  j  } | |  j k r* t d d É Ç n  | d k  rF |  j d } n  |  j j | É } t | É |  j k r t d d É Ç n  | S(   s~    read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. iù  s   Request to largei    i   (   R¢  R≤  R§   R3  Ro  R}   (   RI   t   clenR   (    (    s&   /home/lgardner/git/professor/bottle.pyRú  ö  s    	 c         CÄ  s   |  j  j d É |  j  S(   sl   The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. i    (   R∫  R¥  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR3  ¶  s    c         CÄ  s   d |  j  j d d É j É  k S(   s(    True if Chunked transfer encoding was. R±  t   HTTP_TRANSFER_ENCODINGRï   (   RÁ   Rœ   Rõ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR±  ∞  s    s   bottle.request.postc   	      CÄ  sw  t  É  } |  j j d É s[ t t |  j É  d É É } x | D] \ } } | | | <q= W| Si d d 6} x1 d D]) } | |  j k ro |  j | | | <qo qo Wt d |  j d	 | d
 t	 É } t
 r„ t | d d d d d É| d <n t rˆ d | d <n  t j | ç  } | |  d <| j pg  } xR | D]J } | j r_t | j | j | j | j É | | j <q%| j | | j <q%W| S(   s‹    The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        s
   multipart/R)   Rï   Rê  R⁄   Rô  t   CONTENT_LENGTHt   fpRÁ   t   keep_blank_valuesR(   R=   t   newlines   
s   _cgi.FieldStorage(   s   REQUEST_METHODs   CONTENT_TYPERΩ  (   Rä  t   content_typet
   startswithRë  R©  Rú  RÁ   R]   R3  R©   t   py31RH   R  t   cgit   FieldStorageR[   t   filenameRî  t   fileRì   RÅ  Rm   (	   RI   RZ  Rí  Ra   Rm   t   safe_envR”   R   Rñ  (    (    s&   /home/lgardner/git/professor/bottle.pyRY  ∏  s2    	 
	c         CÄ  s   |  j  j É  S(   sÛ    The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. (   t   urlpartst   geturl(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRŸ   €  s    s   bottle.request.urlpartsc         CÄ  s’   |  j  } | j d É p' | j d d É } | j d É pE | j d É } | sß | j d d É } | j d É } | rß | | d k rä d	 n d
 k rß | d | 7} qß n  t |  j É } t | | | | j d É d É S(   sı    The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. t   HTTP_X_FORWARDED_PROTOs   wsgi.url_schemet   httpt   HTTP_X_FORWARDED_HOSTt	   HTTP_HOSTt   SERVER_NAMEs	   127.0.0.1t   SERVER_PORTt   80t   443t   :Rê  Rï   (   RÁ   Rœ   t   urlquotet   fullpatht   UrlSplitResult(   RI   t   envRÃ  t   hostt   portRä   (    (    s&   /home/lgardner/git/professor/bottle.pyR…  „  s    	!$c         CÄ  s   t  |  j |  j j d É É S(   s:    Request path including :attr:`script_name` (if present). R‹   (   R#   t   script_nameRä   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR’  ı  s    c         CÄ  s   |  j  j d d É S(   sh    The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. Rê  Rï   (   RÁ   Rœ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   query_string˙  s    c         CÄ  s4   |  j  j d d É j d É } | r0 d | d Sd S(   sÒ    The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. RO  Rï   R‹   (   RÁ   Rœ   RP  (   RI   R⁄  (    (    s&   /home/lgardner/git/professor/bottle.pyR⁄     s    i   c         CÄ  s<   |  j  j d d É } t | |  j | É \ |  d <|  d <d S(   s˜    Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        RO  R‹   R€   N(   RÁ   Rœ   R8  Rä   (   RI   t   shiftt   script(    (    s&   /home/lgardner/git/professor/bottle.pyR8  	  s    c         CÄ  s   t  |  j j d É p d É S(   sﬁ    The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. RΩ  iˇˇˇˇ(   Rà   RÁ   Rœ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¢    s    c         CÄ  s   |  j  j d d É j É  S(   sA    The Content-Type header as a lowercase-string (default: empty). Rô  Rï   (   RÁ   Rœ   Rõ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¡    s    c         CÄ  s%   |  j  j d d É } | j É  d k S(   s…    True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). t   HTTP_X_REQUESTED_WITHRï   t   xmlhttprequest(   RÁ   Rœ   Rõ  (   RI   t   requested_with(    (    s&   /home/lgardner/git/professor/bottle.pyt   is_xhr  s    c         CÄ  s   |  j  S(   s9    Alias for :attr:`is_xhr`. "Ajax" is not the right term. (   R·  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   is_ajax'  s    c         CÄ  sK   t  |  j j d d É É } | r% | S|  j j d É } | rG | d f Sd S(   s´   HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. t   HTTP_AUTHORIZATIONRï   t   REMOTE_USERN(   t
   parse_authRÁ   Rœ   Rg   (   RI   t   basict   ruser(    (    s&   /home/lgardner/git/professor/bottle.pyt   auth,  s      
c         CÄ  sa   |  j  j d É } | r> g  | j d É D] } | j É  ^ q( S|  j  j d É } | r] | g Sg  S(   s(   A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. t   HTTP_X_FORWARDED_FORR·   t   REMOTE_ADDR(   RÁ   Rœ   R@  RP  (   RI   t   proxyt   ipt   remote(    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_route:  s
     &c         CÄ  s   |  j  } | r | d Sd S(   sg    The client IP as a string. Note that this information can be forged
            by malicious clients. i    N(   RÓ  Rg   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_addrE  s    	c         CÄ  s   t  |  j j É  É S(   sD    Return a new :class:`Request` with a shallow :attr:`environ` copy. (   t   RequestRÁ   t   copy(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÒ  L  s    c         CÄ  s   |  j  j | | É S(   N(   RÁ   Rœ   (   RI   Rm   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   P  s    c         CÄ  s   |  j  | S(   N(   RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __getitem__Q  s    c         CÄ  s   d |  | <|  j  | =d  S(   NRï   (   RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delitem__R  s   
 c         CÄ  s   t  |  j É S(   N(   Ru  RÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  S  s    c         CÄ  s   t  |  j É S(   N(   R}   RÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __len__T  s    c         CÄ  s   |  j  j É  S(   N(   RÁ   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   U  s    c         CÄ  s¢   |  j  j d É r! t d É Ç n  | |  j  | <d } | d k rI d } n- | d
 k r^ d } n | j d É rv d } n  x% | D] } |  j  j d | d É q} Wd S(   sA    Change an environ value and clear all caches that depend on it. s   bottle.request.readonlys$   The environ dictionary is read-only.s
   wsgi.inputR3  Rï  Rò  Ró  RZ  R†  Rê  R’   t   HTTP_RÅ  Rã  s   bottle.request.N(    (   s   bodys   formss   filess   paramss   posts   json(   s   querys   params(   s   headerss   cookies(   RÁ   Rœ   R“   R¬  R—   Rg   (   RI   Ra   Rm   t   todelete(    (    s&   /home/lgardner/git/professor/bottle.pyt   __setitem__V  s    			c         CÄ  s   d |  j  j |  j |  j f S(   Ns   <%s: %s %s>(   t	   __class__RK   R¥   RŸ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  i  s    c         CÄ  s]   y5 |  j  d | } t | d É r0 | j |  É S| SWn! t k
 rX t d | É Ç n Xd S(   s@    Search in self.environ for additional user defined attributes. s   bottle.request.ext.%sRl   s   Attribute %r not defined.N(   RÁ   R2   Rl   R“   RO   (   RI   Rì   t   var(    (    s&   /home/lgardner/git/professor/bottle.pyt   __getattr__l  s
    $c         CÄ  s4   | d k r t  j |  | | É S| |  j d | <d  S(   NRÁ   s   bottle.request.ext.%s(   t   objectt   __setattr__RÁ   (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸  t  s     N(8   RK   RL   Rp   t	   __slots__R≤  Rg   Rc   R_   R©   RÔ   RA  R•   R  Rä   R¥   RÅ  RÜ  Rã  Rè  R’   Rï  Ró  Rò  R†  Rß  R∞  R∫  Rú  R3  R±  Rﬂ   RY  RŸ   R…  R’  R€  R⁄  R8  R¢  R¡  R·  R‚  RË  RÓ  RÔ  RÒ  Rœ   RÚ  RÛ  Rp  RÙ  R∑   R˜  R  R˙  R¸  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÉ  ﬁ  sd   			
#	
									c         CÄ  s   |  j  É  j d d É S(   NRƒ   t   -(   t   titlet   replace(   R0   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hkey{  s    t   HeaderPropertyc           BÄ  s5   e  Z d e d  d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   Rï   c         CÄ  s=   | | |  _  |  _ | | |  _ |  _ d | j É  |  _ d  S(   Ns   Current value of the %r header.(   Rì   R
   t   readert   writerRˇ  Rp   (   RI   Rì   R  R  R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Ä  s    c         CÄ  sE   | d  k r |  S| j j |  j |  j É } |  j rA |  j | É S| S(   N(   Rg   RÅ  Rœ   Rì   R
   R  (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   Ö  s     c         CÄ  s   |  j  | É | j |  j <d  S(   N(   R  RÅ  Rì   (   RI   Ri   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRn   ä  s    c         CÄ  s   | j  |  j =d  S(   N(   RÅ  Rì   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyRo   ç  s    N(   RK   RL   Rg   Rá   Rc   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR    s   		t   BaseResponsec        
   BÄ  sä  e  Z d  Z d Z d Z i e d+ É d 6e d, É d 6Z d d- d- d Ñ Z d- d Ñ Z	 d Ñ  Z
 d Ñ  Z e d Ñ  É Z e d Ñ  É Z d Ñ  Z d Ñ  Z e e e d- d É Z [ [ e d Ñ  É Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d- d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z e d  Ñ  É Z e d É Z e d d! e ÉZ e d" d! d# Ñ  d$ d% Ñ  ÉZ  e d& d' Ñ É Z! d- d( Ñ Z" d) Ñ  Z# d* Ñ  Z$ RS(.   s∫   Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    i»   s   text/html; charset=UTF-8s   Content-TypeiÃ   R‚   s   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Md5s   Last-Modifiedi0  Rï   c         KÄ  sµ   d  |  _ i  |  _ | |  _ | p' |  j |  _ | r{ t | t É rQ | j É  } n  x' | D] \ } } |  j	 | | É qX Wn  | r± x- | j É  D] \ } } |  j	 | | É qé Wn  d  S(   N(
   Rg   t   _cookiest   _headersR3  t   default_statusR1  R>   R]   t   itemsR2  (   RI   R3  R1  RÅ  t   more_headersRì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ¨  s    			c         CÄ  sì   | p	 t  } t | t  É s! t Ç | É  } |  j | _ t d Ñ  |  j j É  DÉ É | _ |  j rè t É  | _ | j j	 |  j j
 d d É É n  | S(   s    Returns a copy of self. c         sÄ  s"   |  ] \ } } | | f Vq d  S(   N(    (   R√   t   kt   v(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>¿  s    R≠  Rï   (   R  t
   issubclasst   AssertionErrorR1  R]   R  R	  R  R*   RW  t   output(   RI   Rj   RÒ  (    (    s&   /home/lgardner/git/professor/bottle.pyRÒ  ∫  s    	"	"c         CÄ  s   t  |  j É S(   N(   Ru  R3  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ∆  s    c         CÄ  s&   t  |  j d É r" |  j j É  n  d  S(   NRJ   (   R2   R3  RJ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   …  s    c         CÄ  s   |  j  S(   s;    The HTTP status line as a string (e.g. ``404 Not Found``).(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   status_lineÕ  s    c         CÄ  s   |  j  S(   s/    The HTTP status code as an integer (e.g. 404).(   R~  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRr  “  s    c         CÄ  s∂   t  | t É r( | t j | É } } n= d | k rY | j É  } t | j É  d É } n t d É Ç d | k o| d k n sê t d É Ç n  | |  _ t | p© d | É |  _	 d  S(   Nt    i    s+   String status line without a reason phrase.id   iÁ  s   Status code out of range.s
   %d Unknown(
   R>   Rà   t   _HTTP_STATUS_LINESRœ   RP  R@  R£   R~  Rá   R  (   RI   R1  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _set_status◊  s     	c         CÄ  s   |  j  S(   N(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _get_status„  s    sQ   A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. c         CÄ  s   t  É  } |  j | _ | S(   sl    An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. (   t
   HeaderDictR  R]   (   RI   t   hdict(    (    s&   /home/lgardner/git/professor/bottle.pyRÅ  Ó  s    	c         CÄ  s   t  | É |  j k S(   N(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __contains__ˆ  s    c         CÄ  s   |  j  t | É =d  S(   N(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  ˜  s    c         CÄ  s   |  j  t | É d S(   Niˇˇˇˇ(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  ¯  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R  R  (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  ˘  s    c         CÄ  s    |  j  j t | É | g É d S(   s|    Return the value of a previously defined header. If there is no
            header with that name, return a default value. iˇˇˇˇ(   R  Rœ   R  (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRÜ  ˚  s    c         CÄ  s    t  | É g |  j t | É <d S(   sh    Create a new response header, replacing any previously defined
            headers with the same name. N(   Rá   R  R  (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_header   s    c         CÄ  s,   |  j  j t | É g  É j t | É É d S(   s=    Add an additional response header, not removing duplicates. N(   R  R¨   R  R   Rá   (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR2    s    c         CÄ  s   |  j  S(   sx    Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. (   R4  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iter_headers	  s    c   	      CÄ  s  g  } t  |  j j É  É } d |  j k rF | j d |  j g f É n  |  j |  j k rî |  j |  j } g  | D] } | d | k ro | ^ qo } n  | g  | D]% \ } } | D] } | | f ^ qÆ qû 7} |  j r	x3 |  j j É  D] } | j d | j	 É  f É q„ Wn  | S(   s.    WSGI conform list of (header, value) tuples. s   Content-Typei    s
   Set-Cookie(
   R[   R  R	  R   t   default_content_typeR~  t   bad_headersR  Râ  t   OutputString(	   RI   Rx  RÅ  R  t   hRì   t   valst   valRà  (    (    s&   /home/lgardner/git/professor/bottle.pyR4    s    ,6	 R  t   Expiresc         CÄ  s   t  j t |  É É S(   N(   R   t   utcfromtimestampt
   parse_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   !  s    R  c         CÄ  s
   t  |  É S(   N(   t	   http_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   "  s    s   UTF-8c         CÄ  s:   d |  j  k r6 |  j  j d É d j d É d j É  S| S(   sJ    Return the charset specified in the content-type header (default: utf8). s   charset=iˇˇˇˇRö  i    (   R¡  R@  RP  (   RI   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRq  $  s    'c         KÄ  sk  |  j  s t É  |  _  n  | r< t t | | f | É É } n t | t É sZ t d É Ç n  t | É d k r{ t d É Ç n  | |  j  | <x‹ | j	 É  D]Œ \ } } | d k r⁄ t | t
 É r⁄ | j | j d d } q⁄ n  | d k rFt | t t f É r
| j É  } n' t | t t f É r1t j | É } n  t j d | É } n  | |  j  | | j d	 d
 É <qï Wd S(   sπ   Create a new cookie or replace an old one. If the `secret` parameter is
            set, create a `Signed Cookie` (described below).

            :param name: the name of the cookie.
            :param value: the value of the cookie.
            :param secret: a signature key required for signed cookies.

            Additionally, this method accepts all RFC 2109 attributes that are
            supported by :class:`cookie.Morsel`, including:

            :param max_age: maximum age in seconds. (default: None)
            :param expires: a datetime object or UNIX timestamp. (default: None)
            :param domain: the domain that is allowed to read the cookie.
              (default: current domain)
            :param path: limits the cookie to a given path (default: current path)
            :param secure: limit the cookie to HTTPS connections (default: off).
            :param httponly: prevents client-side javascript to read this cookie
              (default: off, requires Python 2.6 or newer).

            If neither `expires` nor `max_age` is set (default), the cookie will
            expire at the end of the browser session (as soon as the browser
            window is closed).

            Signed cookies may store any pickle-able object and are
            cryptographically signed to prevent manipulation. Keep in mind that
            cookies are limited to 4kb in most browsers.

            Warning: Signed cookies are not encrypted (the client can still see
            the content) and not copy-protected (the client can restore an old
            cookie). The main intention is to make pickling and unpickling
            save, not to store secret information at client side.
        s)   Secret key missing for non-string Cookie.i   s   Cookie value to long.t   max_agei   i  t   expiress   %a, %d %b %Y %H:%M:%S GMTRƒ   R˛  N(   R  R*   R/   t   cookie_encodeR>   R?  RJ  R}   R£   R	  R   t   secondst   dayst   datedateR   t	   timetupleRà   Râ   t   timet   gmtimet   strftimeR   (   RI   Rì   Rm   Rç  RC  Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_cookie+  s(    !	 c         KÄ  s+   d | d <d | d <|  j  | d | ç d S(   sq    Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. iˇˇˇˇR$  i    R%  Rï   N(   R.  (   RI   Ra   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete_cookiec  s    

c         CÄ  sD   d } x7 |  j  D], \ } } | d | j É  | j É  f 7} q W| S(   NRï   s   %s: %s
(   R4  Rˇ  RP  (   RI   Rx  Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  j  s    $(   s   Content-Type(   s   Allows   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Types   Content-Md5s   Last-ModifiedN(%   RK   RL   Rp   R  R  R\   R  Rg   Rc   RÒ  Rp  RJ   R  R  Rr  R  R  R1  RÅ  R  RÛ  RÚ  R˜  RÜ  R  R2  R  R4  R  R¡  Rà   R¢  R%  Rq  R.  R/  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  ë  sN    														8	c         Ä  s_   |  r t  d É n  t j É  â  á  f d Ü  } á  f d Ü  } á  f d Ü  } t | | | d É S(   Ns3   local_property() is deprecated and will be removed.c         Ä  s/   y à  j  SWn t k
 r* t d É Ç n Xd  S(   Ns    Request context not initialized.(   R˘  RO   RÑ  (   RI   (   t   ls(    s&   /home/lgardner/git/professor/bottle.pyt   fgett  s     c         Ä  s   | à  _  d  S(   N(   R˘  (   RI   Rm   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fsetx  s    c         Ä  s
   à  `  d  S(   N(   R˘  (   RI   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fdely  s    s   Thread-local property(   RY   t	   threadingt   localR  (   Rì   R1  R2  R3  (    (   R0  s&   /home/lgardner/git/professor/bottle.pyt   local_propertyq  s     t   LocalRequestc           BÄ  s    e  Z d  Z e j Z e É  Z RS(   sT   A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). (   RK   RL   Rp   RÉ  Rc   Rg  R6  RÁ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR7  }  s   	t   LocalResponsec           BÄ  sD   e  Z d  Z e j Z e É  Z e É  Z e É  Z	 e É  Z
 e É  Z RS(   s+   A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    (   RK   RL   Rp   R  Rc   Rg  R6  R  R~  R  R  R3  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  á  s   					R9  c           BÄ  s#   e  Z d  d d d Ñ Z d Ñ  Z RS(   Rï   c         KÄ  s#   t  t |  É j | | | | ç d  S(   N(   t   superR9  Rc   (   RI   R3  R1  RÅ  R
  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ö  s    c         CÄ  s@   |  j  | _  |  j | _ |  j | _ |  j | _ |  j | _ d  S(   N(   R~  R  R  R  R3  (   RI   Rh  (    (    s&   /home/lgardner/git/professor/bottle.pyR˙   ù  s
    N(   RK   RL   Rg   Rc   R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR9  ô  s   R§   c           BÄ  s#   e  Z d  Z d d d d d Ñ Z RS(   iÙ  c         KÄ  s2   | |  _  | |  _ t t |  É j | | | ç d  S(   N(   t	   exceptiont	   tracebackR9  R§   Rc   (   RI   R1  R3  R:  R;  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ß  s    		N(   RK   RL   R  Rg   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR§   •  s   t   PluginErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR<  µ  s    R!  c           BÄ  s)   e  Z d  Z d Z e d Ñ Z d Ñ  Z RS(   R†  i   c         CÄ  s   | |  _  d  S(   N(   R   (   RI   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   º  s    c         Ä  s)   |  j  â à s à  Sá  á f d Ü  } | S(   Nc          Ä  sõ   y à  |  | é  } Wn t  k
 r/ t É  } n Xt | t É rX à | É } d t _ | St | t É ró t | j t É ró à | j É | _ d | _ n  | S(   Ns   application/json(   R§   R   R>   R]   Rh  R¡  R9  R3  (   R4   RR   t   rvt   json_response(   R   R   (    s&   /home/lgardner/git/professor/bottle.pyRP   ¬  s    	!(   R   (   RI   R   RA  RP   (    (   R   R   s&   /home/lgardner/git/professor/bottle.pyR˙   ø  s
    	 (   RK   RL   Rì   R  R   Rc   R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR!  ∏  s   R"  c           BÄ  s#   e  Z d  Z d Z d Z d Ñ  Z RS(   s   This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. Rb  i   c         CÄ  s{   | j  j d É } t | t t f É rT t | É d k rT t | d | d ç | É St | t É rs t | É | É S| Sd  S(   NRb  i   i    i   (   Rı   Rœ   R>   RZ   R[   R}   t   viewRá   (   RI   R   RA  RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙   ﬁ  s    '(   RK   RL   Rp   Rì   R  R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR"  ÷  s   t   _ImportRedirectc           BÄ  s&   e  Z d  Ñ  Z d d Ñ Z d Ñ  Z RS(   c         CÄ  sv   | |  _  | |  _ t j j | t j | É É |  _ |  j j j	 i t
 d 6g  d 6g  d 6|  d 6É t j j |  É d S(   s@    Create a virtual package that redirects imports (see PEP 302). t   __file__t   __path__t   __all__t
   __loader__N(   Rì   t   impmaskR   t   modulesR¨   t   impt
   new_modulet   moduleRs   t   updateRA  t	   meta_pathR   (   RI   Rì   RE  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Í  s    		!c         CÄ  s=   d | k r d  S| j  d d É d } | |  j k r9 d  S|  S(   Nt   .i   i    (   t   rsplitRì   (   RI   t   fullnameRä   t   packname(    (    s&   /home/lgardner/git/professor/bottle.pyt   find_moduleÛ  s      c         CÄ  s   | t  j k r t  j | S| j d d É d } |  j | } t | É t  j | } t  j | <t |  j | | É |  | _ | S(   NRL  i   (   R   RF  RM  RE  t
   __import__Ru   RI  RD  (   RI   RN  t   modnamet   realnameRI  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_module˘  s     
	N(   RK   RL   Rc   Rg   RP  RT  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR@  È  s   		t	   MultiDictc           BÄ  s
  e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 e rê d	 Ñ  Z d
 Ñ  Z d Ñ  Z e
 Z e Z e Z e Z n? d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d d d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z e Z e Z RS(   sÈ    This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    c         OÄ  s,   t  d Ñ  t  | | é  j É  DÉ É |  _  d  S(   Nc         sÄ  s$   |  ] \ } } | | g f Vq d  S(   N(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R	  (   RI   R4   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s    c         CÄ  s   t  |  j É S(   N(   R}   R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ    s    c         CÄ  s   t  |  j É S(   N(   Ru  R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp    s    c         CÄ  s   | |  j  k S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s    c         CÄ  s   |  j  | =d  S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ    s    c         CÄ  s   |  j  | d S(   Niˇˇˇˇ(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ    s    c         CÄ  s   |  j  | | É d  S(   N(   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜    s    c         CÄ  s   |  j  j É  S(   N(   R]   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR∑     s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s   |  ] } | d  Vq d S(   iˇˇˇˇN(    (   R√   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   Râ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRâ    s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s%   |  ] \ } } | | d  f Vq d S(   iˇˇˇˇN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>   s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR	     s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R√   R  t   vlR  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>"  s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRì  !  s    c         CÄ  s$   g  |  j  j É  D] } | d ^ q S(   Niˇˇˇˇ(   R]   Râ  (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRâ  )  s    c         CÄ  s0   g  |  j  j É  D] \ } } | | d f ^ q S(   Niˇˇˇˇ(   R]   R	  (   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  *  s    c         CÄ  s   |  j  j É  S(   N(   R]   t   iterkeys(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRW  +  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s   |  ] } | d  Vq d S(   iˇˇˇˇN(    (   R√   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>,  s    (   R]   t
   itervalues(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRX  ,  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s%   |  ] \ } } | | d  f Vq d S(   iˇˇˇˇN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>.  s    (   R]   t	   iteritems(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRY  -  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R√   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>0  s    (   R]   RY  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iterallitems/  s    c         CÄ  s9   g  |  j  j É  D]% \ } } | D] } | | f ^ q  q S(   N(   R]   RY  (   RI   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRì  1  s    iˇˇˇˇc         CÄ  sA   y) |  j  | | } | r$ | | É S| SWn t k
 r< n X| S(   s”   Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        (   R]   Rm  (   RI   Ra   R
   t   indexR˛   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   4  s    
c         CÄ  s    |  j  j | g  É j | É d S(   s5    Add a new value to the list of values for this key. N(   R]   R¨   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   E  s    c         CÄ  s   | g |  j  | <d S(   s1    Replace the list of values with a single value. N(   R]   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   I  s    c         CÄ  s   |  j  j | É p g  S(   s5    Return a (possibly empty) list of values for a key. (   R]   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   getallM  s    N(   RK   RL   Rp   Rc   RÙ  Rp  R  RÛ  RÚ  R˜  R∑   R  Râ  R	  Rì  RW  RX  RY  RZ  Rg   Rœ   R   R   R\  t   getonet   getlist(    (    (    s&   /home/lgardner/git/professor/bottle.pyRU    s<   																						Rä  c           BÄ  sP   e  Z d  Z d Z e Z d d Ñ Z d d Ñ Z d d d Ñ Z	 e
 É  d Ñ Z RS(   s©   This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. R=   c         CÄ  sd   t  | t É r7 |  j r7 | j d É j | p3 |  j É St  | t É r\ | j | pX |  j É S| Sd  S(   NR)   (   R>   R?   t   recode_unicodeR@   RE   t   input_encodingRA   (   RI   R0   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _fixd  s
    c         CÄ  sq   t  É  } | p |  j } | _ t | _ xB |  j É  D]4 \ } } | j |  j | | É |  j | | É É q5 W| S(   s™    Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. (   Rä  R`  Rq   R_  Rì  R   Ra  (   RI   R(   RÒ  RB   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRE   l  s    		,c         CÄ  s7   y |  j  |  | | É SWn t t f k
 r2 | SXd S(   s7    Return the value as a unicode string, or the default. N(   Ra  Rf  R“   (   RI   Rì   R
   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   getunicodew  s    c         CÄ  sG   | j  d É r4 | j d É r4 t t |  É j | É S|  j | d | ÉS(   Nt   __R
   (   R¬  RB  R9  Rä  R˙  Rb  (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  ~  s    N(   RK   RL   Rp   R`  R©   R_  Rg   Ra  RE   Rb  R?   R˙  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRä  V  s   R  c           BÄ  sn   e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d d	 d
 Ñ Z d Ñ  Z RS(   sz    A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. c         OÄ  s,   i  |  _  | s | r( |  j | | é  n  d  S(   N(   R]   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   â  s    	 c         CÄ  s   t  | É |  j k S(   N(   R  R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ç  s    c         CÄ  s   |  j  t | É =d  S(   N(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  é  s    c         CÄ  s   |  j  t | É d S(   Niˇˇˇˇ(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  è  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  ê  s    c         CÄ  s,   |  j  j t | É g  É j t | É É d  S(   N(   R]   R¨   R  R   Rá   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   ë  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   ì  s    c         CÄ  s   |  j  j t | É É p g  S(   N(   R]   Rœ   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR\  î  s    iˇˇˇˇc         CÄ  s   t  j |  t | É | | É S(   N(   RU  Rœ   R  (   RI   Ra   R
   R[  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   ï  s    c         CÄ  sJ   xC g  | D] } t  | É ^ q
 D]" } | |  j k r  |  j | =q  q  Wd  S(   N(   R  R]   (   RI   t   namesR◊   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   filteró  s    &N(   RK   RL   Rp   Rc   R  RÛ  RÚ  R˜  R   R   R\  Rg   Rœ   Re  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  Ö  s   								RÖ  c           BÄ  sq   e  Z d  Z d Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z RS(   s    This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    Rô  RΩ  c         CÄ  s   | |  _  d  S(   N(   RÁ   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ´  s    c         CÄ  s3   | j  d d É j É  } | |  j k r+ | Sd | S(   s6    Translate header field name to CGI/WSGI environ key. R˛  Rƒ   Rı  (   R   R„   t   cgikeys(   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _ekeyÆ  s    c         CÄ  s   |  j  j |  j | É | É S(   s:    Return the header value as is (may be bytes or unicode). (   RÁ   Rœ   Rg  (   RI   Ra   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt   rawµ  s    c         CÄ  s   t  |  j |  j | É d É S(   NR)   (   R©  RÁ   Rg  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  π  s    c         CÄ  s   t  d |  j É Ç d  S(   Ns   %s is read-only.(   RJ  R¯  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  º  s    c         CÄ  s   t  d |  j É Ç d  S(   Ns   %s is read-only.(   RJ  R¯  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  ø  s    c         cÄ  so   xh |  j  D]] } | d  d k r> | d j d d É j É  Vq
 | |  j k r
 | j d d É j É  Vq
 q
 Wd  S(   Ni   Rı  Rƒ   R˛  (   RÁ   R   Rˇ  Rf  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ¬  s
    c         CÄ  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   …  s    c         CÄ  s   t  |  j É  É S(   N(   R}   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ     s    c         CÄ  s   |  j  | É |  j k S(   N(   Rg  RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  À  s    (   s   CONTENT_TYPEs   CONTENT_LENGTHN(   RK   RL   Rp   Rf  Rc   Rg  Rg   Rh  RÚ  R˜  RÛ  Rp  R∑   RÙ  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÖ  ù  s   
								RÛ   c           BÄ  s∫   e  Z d  Z d Z d e f d Ñ  É  YZ d Ñ  Z d Ñ  Z d e d Ñ Z	 d	 Ñ  Z
 d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   sH   A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    t   _metaR  t	   Namespacec           BÄ  sÜ   e  Z d  Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   c         CÄ  s   | |  _  | |  _ d  S(   N(   t   _configt   _prefix(   RI   Rı   t	   namespace(    (    s&   /home/lgardner/git/professor/bottle.pyRc   €  s    	c         CÄ  s    t  d É |  j |  j d | S(   Ns}   Accessing namespaces as dicts is discouraged. Only use flat item access: cfg["names"]["pace"]["key"] -> cfg["name.space.key"]RL  (   RY   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  ﬂ  s    
c         CÄ  s   | |  j  |  j d | <d  S(   NRL  (   Rk  Rl  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  Â  s    c         CÄ  s   |  j  |  j d | =d  S(   NRL  (   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  Ë  s    c         cÄ  sZ   |  j  d } xF |  j D]; } | j d É \ } } } | |  j  k r | r | Vq q Wd  S(   NRL  (   Rl  Rk  t
   rpartition(   RI   t	   ns_prefixRa   t   nst   dotRì   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  Î  s
    c         CÄ  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   Ú  s    c         CÄ  s   t  |  j É  É S(   N(   R}   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ  Û  s    c         CÄ  s   |  j  d | |  j k S(   NRL  (   Rl  Rk  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  Ù  s    c         CÄ  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ı  s    c         CÄ  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __str__ˆ  s    c         CÄ  sÑ   t  d É | |  k rM | d j É  rM t j |  j |  j d | É |  | <n  | |  k rw | j d É rw t | É Ç n  |  j | É S(   Ns   Attribute access is deprecated.i    RL  Rc  (	   RY   t   isupperRÛ   Rj  Rk  Rl  R¬  RO   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  ˘  s    
'c         CÄ  sé   | d k r | |  j  | <d  St d É t t | É rE t d É Ç n  | |  k rÄ |  | rÄ t |  | |  j É rÄ t d É Ç n  | |  | <d  S(   NRk  Rl  s#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   s   _configs   _prefix(   Rs   RY   R2   R9   RO   R>   R¯  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸    s    
,c         CÄ  so   | |  k rk |  j  | É } t | |  j É rk | d } x. |  D]# } | j | É r> |  | | =q> q> Wqk n  d  S(   NRL  (   R—   R>   R¯  R¬  (   RI   Ra   R  Rù   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delattr__  s    
c         OÄ  s   t  d É |  j | | é  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1     s    
(   RK   RL   Rc   RÚ  R˜  RÛ  Rp  R∑   RÙ  R  R  Rr  R˙  R¸  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRj  Ÿ  s   														c         OÄ  sB   i  |  _  d Ñ  |  _ | s! | r> t d É |  j | | é  n  d  S(   Nc         SÄ  s   d  S(   N(   Rg   (   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    s-   Constructor does no longer accept parameters.(   Ri  R  RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s
    	
c         CÄ  sx   t  É  } | j | É x[ | j É  D]M } xD | j | É D]3 \ } } | d k rb | d | } n  | |  | <q9 Wq# W|  S(   s   Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        t   DEFAULTt   bottleRL  (   Ru  s   bottle(   R-   Ro  t   sectionsR	  (   RI   R∆  RÜ   t   sectionRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_config!  s    	Rï   c   	      CÄ  s  | | f g } xÒ | r| j  É  \ } } t | t É sR t d t | É É Ç n  x™ | j É  D]ú \ } } t | t É sì t d t | É É Ç n  | rß | d | n | } t | t É rÒ | j | | f É | r˚ |  j |  | É |  | <q˚ q_ | |  | <q_ Wq W|  S(   s‰    Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        s   Source is not a dict (r)s   Key is not a string (%r)RL  (	   R—   R>   R]   RJ  R˛   R	  R?  R   Rj  (	   RI   t   sourceRm  RÓ   t   stackRù   Ra   Rm   t   full_key(    (    s&   /home/lgardner/git/professor/bottle.pyRÙ   1  s    	c         OÄ  s{   d } | rC t  | d t É rC | d j d É d } | d } n  x1 t | | é  j É  D] \ } } | |  | | <qY Wd S(   s’    If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` Rï   i    RL  i   N(   R>   R?  RP  R]   R	  (   RI   R4   RR   Rù   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ  I  s    "c         CÄ  s!   | |  k r | |  | <n  |  | S(   N(    (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¨   T  s    c         CÄ  sç   t  | t É s( t d t | É É Ç n  |  j | d d Ñ  É | É } | |  k rf |  | | k rf d  S|  j | | É t j |  | | É d  S(   Ns   Key has type %r (not a string)Re  c         SÄ  s   |  S(   N(    (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]  s    (   R>   R?  RJ  R˛   t   meta_getR  R]   R˜  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  Y  s    c         CÄ  s   t  j |  | É d  S(   N(   R]   RÛ  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  c  s    c         CÄ  s   x |  D] } |  | =q Wd  S(   N(    (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   clearf  s    c         CÄ  s   |  j  j | i  É j | | É S(   s-    Return the value of a meta field for a key. (   Ri  Rœ   (   RI   Ra   t	   metafieldR
   (    (    s&   /home/lgardner/git/professor/bottle.pyR}  j  s    c         CÄ  s:   | |  j  j | i  É | <| |  k r6 |  | |  | <n  d S(   sq    Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. N(   Ri  R¨   (   RI   Ra   R  Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  n  s    c         CÄ  s   |  j  j | i  É j É  S(   s;    Return an iterable of meta field names defined for a key. (   Ri  Rœ   R∑   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   meta_listu  s    c         CÄ  sv   t  d É | |  k r? | d j É  r? |  j |  | É |  | <n  | |  k ri | j d É ri t | É Ç n  |  j | É S(   Ns   Attribute access is deprecated.i    Rc  (   RY   Rs  Rj  R¬  RO   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  z  s    
c         CÄ  sì   | |  j  k r" t j |  | | É St d É t t | É rJ t d É Ç n  | |  k rÖ |  | rÖ t |  | |  j É rÖ t d É Ç n  | |  | <d  S(   Ns#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   R˝  R]   R¸  RY   R2   RO   R>   Rj  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸  Ç  s    
,c         CÄ  so   | |  k rk |  j  | É } t | |  j É rk | d } x. |  D]# } | j | É r> |  | | =q> q> Wqk n  d  S(   NRL  (   R—   R>   Rj  R¬  (   RI   Ra   R  Rù   (    (    s&   /home/lgardner/git/professor/bottle.pyRt  å  s    
c         OÄ  s   t  d É |  j | | é  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ï  s    
(   s   _metas
   _on_changeN(   RK   RL   Rp   R˝  R9   Rj  Rc   Ry  Rq   RÙ   RJ  R¨   R˜  RÛ  R~  Rg   R}  R  RÄ  R˙  R¸  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ   œ  s$   A					
						
		t   AppStackc           BÄ  s#   e  Z d  Z d Ñ  Z d d Ñ Z RS(   s>    A stack-like list. Calling it returns the head of the stack. c         CÄ  s   |  d S(   s)    Return the current default application. iˇˇˇˇ(    (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ü  s    c         CÄ  s,   t  | t É s t É  } n  |  j | É | S(   s1    Add a new :class:`Bottle` instance to the stack (   R>   R  R   (   RI   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   push£  s    N(   RK   RL   Rp   R1   Rg   RÇ  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÅ  ú  s   	Rt  c           BÄ  s   e  Z d d Ñ Z d Ñ  Z RS(   i   i@   c         CÄ  sS   | | |  _  |  _ x9 d D]1 } t | | É r t |  | t | | É É q q Wd  S(   Nt   filenoRJ   Ro  t	   readlinest   tellR¥  (   s   filenos   closes   reads	   readliness   tells   seek(   Ræ  t   buffer_sizeR2   Ru   Rh   (   RI   Ræ  RÜ  R`   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ≠  s     c         cÄ  s?   |  j  |  j } } x% t r: | | É } | s2 d  S| Vq Wd  S(   N(   RÜ  Ro  R©   (   RI   RØ  Ro  R¶  (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ≤  s    	 i   (   RK   RL   Rc   Rp  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt  ´  s   Rw  c           BÄ  s,   e  Z d  Z d d Ñ Z d Ñ  Z d Ñ  Z RS(   sä    This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). c         CÄ  s   | |  _  t | É |  _ d  S(   N(   t   iteratorR^   t   close_callbacks(   RI   Rá  RJ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   æ  s    	c         CÄ  s   t  |  j É S(   N(   Ru  Rá  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ¬  s    c         CÄ  s   x |  j  D] } | É  q
 Wd  S(   N(   Rà  (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   ≈  s    N(   RK   RL   Rp   Rg   Rc   Rp  RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw  ∫  s   	R  c           BÄ  sP   e  Z d  Z d e d d Ñ Z d	 d	 e d Ñ Z d Ñ  Z d Ñ  Z	 d d Ñ Z RS(
   sf   This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    s   ./t   allc         CÄ  s1   t  |  _ | |  _ | |  _ g  |  _ i  |  _ d  S(   N(   t   opent   openert   baset	   cachemodeRä   t   cache(   RI   Rå  Rã  Rç  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ‘  s
    				c         CÄ  s˚   t  j j t  j j | p |  j É É } t  j j t  j j | t  j j | É É É } | t  j 7} | |  j k rÉ |  j j | É n  | r¨ t  j j | É r¨ t  j	 | É n  | d k rÀ |  j j | É n |  j j | | É |  j j É  t  j j | É S(   s   Add a new path to the list of search paths. Return False if the
            path does not exist.

            :param path: The new search path. Relative paths are turned into
                an absolute and normalized form. If the path looks like a file
                (not ending in `/`), the filename is stripped off.
            :param base: Path used to absolutize relative search paths.
                Defaults to :attr:`base` which defaults to ``os.getcwd()``.
            :param index: Position within the list of search paths. Defaults
                to last index (appends to the list).

            The `base` parameter makes it easy to reference files installed
            along with a python module or package::

                res.add_path('./resources/', __file__)
        N(   t   osRä   t   abspatht   dirnameRå  R»   t   sepR+  t   isdirt   makedirsRg   R   R)  Ré  R~  t   exists(   RI   Rä   Rå  R[  t   create(    (    s&   /home/lgardner/git/professor/bottle.pyt   add_pathﬁ  s    '-c         cÄ  sï   |  j  } xÑ | rê | j É  } t j  j | É s7 q n  xS t j | É D]B } t j  j | | É } t j  j | É rÑ | j | É qG | VqG Wq Wd S(   s:    Iterate over all existing files in all registered paths. N(   Rä   R—   Rè  Rì  t   listdirR»   R   (   RI   t   searchRä   Rì   t   full(    (    s&   /home/lgardner/git/professor/bottle.pyRp  ˝  s    
	  c         CÄ  s†   | |  j  k s t rï x[ |  j D]P } t j j | | É } t j j | É r |  j d k rk | |  j  | <n  | Sq W|  j d k rï d |  j  | <qï n  |  j  | S(   s˙    Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. Râ  t   found(   s   alls   foundN(   Ré  R±   Rä   Rè  R»   t   isfileRç  Rg   (   RI   Rì   Rä   t   fpath(    (    s&   /home/lgardner/git/professor/bottle.pyt   lookup	  s    t   rc         OÄ  sA   |  j  | É } | s( t d | É Ç n  |  j | d | | | éS(   s=    Find a resource and return a file object, or raise IOError. s   Resource %r not found.R∫   (   Rû  t   IOErrorRã  (   RI   Rì   R∫   R”   R.  t   fname(    (    s&   /home/lgardner/git/professor/bottle.pyRä  	  s     N(
   RK   RL   Rp   Rä  Rc   Rg   Rq   Ró  Rp  Rû  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR     s   
		Rî  c           BÄ  sb   e  Z d d  Ñ Z e d É Z e d d e d d ÉZ e d Ñ  É Z	 d d	 Ñ Z
 e d d
 Ñ Z RS(   c         CÄ  s=   | |  _  | |  _ | |  _ | r- t | É n t É  |  _ d S(   s    Wrapper for file uploads. N(   R«  Rì   t   raw_filenameR  RÅ  (   RI   t   fileobjRì   R∆  RÅ  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   "	  s    			s   Content-Types   Content-LengthR  R
   iˇˇˇˇc         CÄ  sº   |  j  } t | t É s- | j d d É } n  t d | É j d d É j d É } t j j | j	 d t j j
 É É } t j d d | É j É  } t j d d	 | É j d
 É } | d  pª d S(   s—   Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        R=   t   ignoret   NFKDt   ASCIIs   \s   [^a-zA-Z0-9-_.\s]Rï   s   [-\s]+R˛  s   .-iˇ   t   empty(   R¢  R>   R?   RE   R   R@   Rè  Rä   t   basenameR   Rí  RÄ   RÅ   RP  (   RI   R°  (    (    s&   /home/lgardner/git/professor/bottle.pyR∆  0	  s    
	$$i   i   c         CÄ  sa   |  j  j | j |  j  j É  } } } x$ | | É } | s? Pn  | | É q) W|  j  j | É d  S(   N(   R«  Ro  R   RÖ  R¥  (   RI   Ræ  t
   chunk_sizeRo  R   Rú   t   buf(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _copy_fileC	  s    & c         CÄ  s£   t  | t É rè t j j | É r< t j j | |  j É } n  | rd t j j | É rd t d É Ç n  t	 | d É è } |  j
 | | É Wd QXn |  j
 | | É d S(   sÃ   Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        s   File exists.t   wbN(   R>   R?  Rè  Rä   Rì  R»   R∆  Rï  R†  Rä  R´  (   RI   t   destinationt	   overwriteR©  Ræ  (    (    s&   /home/lgardner/git/professor/bottle.pyt   saveK	  s    	Ni   i   (   RK   RL   Rg   Rc   R  R¡  Rà   R¢  Rr   R∆  R´  Rq   RØ  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRî   	  s   iÙ  s   Unknown Error.c         CÄ  s   t  |  | É Ç d S(   s+    Aborts execution and causes a HTTP error. N(   R§   (   R`  t   text(    (    s&   /home/lgardner/git/professor/bottle.pyt   aborth	  s    c         CÄ  st   | s* t  j d É d k r! d n d } n  t j d t É } | | _ d | _ | j d t t  j	 |  É É | Ç d S(	   sd    Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. t   SERVER_PROTOCOLs   HTTP/1.1i/  i.  Rj   Rï   t   LocationN(
   R7  Rœ   Rh  RÒ  R9  R1  R3  R  R#   RŸ   (   RŸ   R`  Rd  (    (    s&   /home/lgardner/git/professor/bottle.pyt   redirectm	  s    $		i   c         cÄ  s[   |  j  | É xG | d k rV |  j t | | É É } | s> Pn  | t | É 8} | Vq Wd S(   sF    Yield chunks from a range in a file. No chunk is bigger than maxread.i    N(   R¥  Ro  R£  R}   (   Ræ  Rú   RA   R•  R¶  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _file_iter_rangey	  s     t   autos   UTF-8c         CÄ  s@  t  j j | É t  j } t  j j t  j j | |  j d É É É }  t É  } |  j | É sh t d d É St  j j	 |  É sé t  j j
 |  É rõ t d d É St  j |  t  j É sΩ t d d É S| d k rÙ t j |  É \ } } | rÙ | | d <qÙ n  | r:| d	  d
 k r-| r-d | k r-| d | 7} n  | | d <n  | rut  j j | t k r[|  n | É } d | | d <n  t  j |  É } | j | d <} t j d t j | j É É }	 |	 | d <t j j d É }
 |
 r˜t |
 j d É d j É  É }
 n  |
 d% k	 rD|
 t | j É k rDt j d t j É  É | d <t d d | ç St j d k rYd n t  |  d É } d | d <t j j d É } d t j k r3t! t" t j d | É É } | s¬t d d  É S| d \ } } d! | | d" | f | d# <t# | | É | d <| r t$ | | | | É } n  t | d d$ | çSt | | ç S(&   sŸ   Open a file in a safe way and return :exc:`HTTPResponse` with status
        code 200, 305, 403 or 404. The ``Content-Type``, ``Content-Encoding``,
        ``Content-Length`` and ``Last-Modified`` headers are set if possible.
        Special support for ``If-Modified-Since``, ``Range`` and ``HEAD``
        requests.

        :param filename: Name or path of the file to send.
        :param root: Root path for file lookups. Should be an absolute directory
            path.
        :param mimetype: Defines the content-type header (default: guess from
            file extension)
        :param download: If True, ask the browser to open a `Save as...` dialog
            instead of opening the file with the associated program. You can
            specify a custom filename as a string. If not specified, the
            original filename is used (default: False).
        :param charset: The charset to use for files with a ``text/*``
            mime-type. (default: UTF-8)
    s   /\iì  s   Access denied.iî  s   File does not exist.s/   You do not have permission to access this file.R∂  s   Content-Encodingi   s   text/Rq  s   ; charset=%ss   Content-Types   attachment; filename="%s"s   Content-Dispositions   Content-Lengths   %a, %d %b %Y %H:%M:%S GMTs   Last-Modifiedt   HTTP_IF_MODIFIED_SINCERö  i    t   DateR1  i0  R›   Rï   t   rbRA   s   Accept-Rangest
   HTTP_RANGEi†  s   Requested Range Not Satisfiables   bytes %d-%d/%di   s   Content-RangeiŒ   N(%   Rè  Rä   Rê  Rí  R»   RP  R]   R¬  R§   Rï  Rú  t   accesst   R_OKt	   mimetypest
   guess_typeR®  R©   t   statt   st_sizeR+  R-  R,  t   st_mtimeR7  RÁ   Rœ   R"  R@  Rg   Rà   R9  R¥   Rä  R[   t   parse_range_headerRá   Rµ  (   R∆  t   roott   mimetypet   downloadRq  RÅ  R(   t   statsRª  t   lmt   imsR3  t   rangesRú   Rö   (    (    s&   /home/lgardner/git/professor/bottle.pyt   static_fileÉ	  sX    *	& "$
"!$
 c         CÄ  s&   |  r t  j d É n  t |  É a d S(   sS    Change the debug level.
    There is only one debug level supported at the moment.R
   N(   RT   t   simplefilterR  R±   (   R∫   (    (    s&   /home/lgardner/git/professor/bottle.pyt   debug‘	  s     c         CÄ  ss   t  |  t t f É r$ |  j É  }  n' t  |  t t f É rK t j |  É }  n  t  |  t É so t j	 d |  É }  n  |  S(   Ns   %a, %d %b %Y %H:%M:%S GMT(
   R>   R)  R   t   utctimetupleRà   Râ   R+  R,  R?  R-  (   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR#  €	  s    c         CÄ  se   y@ t  j j |  É } t j | d  d É | d p6 d t j SWn t t t t	 f k
 r` d SXd S(   sD    Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. i   i    i	   N(   i    (   t   emailt   utilst   parsedate_tzR+  t   mktimet   timezoneRJ  R£   t
   IndexErrort   OverflowErrorRg   (   R»  t   ts(    (    s&   /home/lgardner/git/professor/bottle.pyR"  ‰	  s
    .c         CÄ  sÑ   ye |  j  d d É \ } } | j É  d k rd t t j t | É É É j  d d É \ } } | | f SWn t t f k
 r d SXd S(   s]    Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or Nonei   RÊ  R”  N(	   R@  Rg   Rõ  R/   t   base64t	   b64decodeRC   R“   R£   (   R≠  R¥   R   t   usert   pwd(    (    s&   /home/lgardner/git/professor/bottle.pyRÂ  Ï	  s    -c         cÄ  s,  |  s |  d  d k r d Sg  |  d j  d É D]$ } d | k r/ | j  d d É ^ q/ } xÃ | D]ƒ \ } } y§ | sò t d | t | É É | } } nB | s¥ t | É | } } n& t | É t t | É d | É } } d | k o¸ | k  o¸ | k n r| | f Vn  Wq` t k
 r#q` Xq` Wd S(   s~    Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.i   s   bytes=NR·   R˛  i   i    (   R@  R°  Rà   R£  R£   (   R≠  t   maxlenRü  R…  Rò   Rö   (    (    s&   /home/lgardner/git/professor/bottle.pyR¬  ˆ	  s     >#&'c         CÄ  sª   g  } xÆ |  j  d d É j d É D]ë } | s4 q" n  | j d d É } t | É d k rh | j d É n  t | d j  d d	 É É } t | d j  d d	 É É } | j | | f É q" W| S(
   NRö  t   &t   =i   i   Rï   i    t   +R  (   R   R@  R}   R   t
   urlunquote(   t   qsRü  t   pairt   nvRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRë  
  s    "  c         CÄ  s6   t  d Ñ  t |  | É DÉ É o5 t |  É t | É k S(   ss    Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. c         sÄ  s-   |  ]# \ } } | | k r! d  n d Vq d S(   i    i   N(    (   R√   R    t   y(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>
  s    (   t   sumt   zipR}   (   R4   Rü  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _lscmp
  s    c         CÄ  s^   t  j t j |  d É É } t  j t j t | É | É j É  É } t d É | t d É | S(   s>    Encode and sign a pickle-able object. Return a (byte) string iˇˇˇˇt   !RŒ   (   R÷  t	   b64encodet   pickleR   t   hmact   newRC   t   digest(   R   Ra   R¡   t   sig(    (    s&   /home/lgardner/git/professor/bottle.pyR&  
  s    'c         CÄ  sá   t  |  É }  t |  É rÉ |  j t  d É d É \ } } t | d t j t j t  | É | É j É  É É rÉ t	 j
 t j | É É Sn  d S(   s?    Verify and decode an encoded string. Return an object or None.RŒ   i   N(   RC   t   cookie_is_encodedR@  RÂ  R÷  RÁ  RÈ  RÍ  RÎ  RË  R   R◊  Rg   (   R   Ra   RÏ  R¡   (    (    s&   /home/lgardner/git/professor/bottle.pyRå   
  s    4c         CÄ  s+   t  |  j t d É É o' t d É |  k É S(   s9    Return True if the argument looks like a encoded cookie.RÊ  RŒ   (   R  R¬  RC   (   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRÌ  *
  s    c         CÄ  s@   |  j  d d É j  d d É j  d d É j  d d É j  d	 d
 É S(   s;    Escape HTML special characters ``&<>`` and quotes ``'"``. R€  s   &amp;t   <s   &lt;t   >s   &gt;t   "s   &quot;t   's   &#039;(   R   (   t   string(    (    s&   /home/lgardner/git/professor/bottle.pyRÄ  /
  s    *c         CÄ  s2   d t  |  É j d d É j d d É j d d É S(   s;    Escape and quote a string to be used as an HTTP attribute.s   "%s"s   
s   &#10;s   s   &#13;s   	s   &#9;(   RÄ  R   (   RÚ  (    (    s&   /home/lgardner/git/professor/bottle.pyt
   html_quote5
  s    c         cÄ  sß   d |  j  j d d É j d É } t |  É } t | d É t | d pK g  É } | d | t | d |  É 7} | Vx) | d | D] } | d | 7} | VqÜ Wd S(   sì   Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    R‹   Rc  i    i   s   /<%s>N(   RK   R   RQ  R   R}   RZ   (   Rf   Rä   t   spect   argct   arg(    (    s&   /home/lgardner/git/professor/bottle.pyRX  ;
  s    
"$ c   	      CÄ  s}  | d k r |  | f S| j  d É j d É } |  j  d É j d É } | re | d d k re g  } n  | rÑ | d d k rÑ g  } n  | d k r√ | t | É k r√ | |  } | | } | | } nh | d k  r| t | É k r| | } | | } | |  } n( | d k  rd n d } t d | É Ç d d j | É } d d j | É } | j d É rs| rs| d 7} n  | | f S(   sS   Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    i    R‹   Rï   RO  R€   s"   Cannot shift. Nothing left from %s(   RP  R@  R}   R  R»   RB  (	   R⁄  t	   path_infoR‹  t   pathlistt
   scriptlistt   movedRß  t   new_script_namet   new_path_info(    (    s&   /home/lgardner/git/professor/bottle.pyR8  O
  s.    	 
 	 	



 t   privates   Access deniedc         Ä  s   á  á á f d Ü  } | S(   se    Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. c         Ä  s   á á  á á f d Ü  } | S(   Nc          Ä  se   t  j p d \ } } | d  k s1 à  | | É rX t d à É } | j d d à É | Sà |  | é  S(   Nië  s   WWW-Authenticates   Basic realm="%s"(   NN(   R7  RË  Rg   R§   R2  (   R4   RR   Rÿ  t   passwordRF   (   t   checkRf   t   realmR∞  (    s&   /home/lgardner/git/professor/bottle.pyRP   r
  s    (    (   Rf   RP   (   Rˇ  R   R∞  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  q
  s    (    (   Rˇ  R   R∞  R0  (    (   Rˇ  R   R∞  s&   /home/lgardner/git/professor/bottle.pyt
   auth_basicn
  s    	c         Ä  s+   t  j t t à  É É á  f d Ü  É } | S(   sA    Return a callable that relays calls to the current default app. c          Ä  s   t  t É  à  É |  | é  S(   N(   Rh   RÔ   (   R4   RR   (   Rì   (    s&   /home/lgardner/git/professor/bottle.pyRP   Ç
  s    (   RM   t   wrapsRh   R  (   Rì   RP   (    (   Rì   s&   /home/lgardner/git/professor/bottle.pyt   make_default_app_wrapperÄ
  s    'RA  Rœ   RZ  R\  R^  RØ   RE  R/  R   RL  RV  t   ServerAdapterc           BÄ  s/   e  Z e Z d  d d Ñ Z d Ñ  Z d Ñ  Z RS(   s	   127.0.0.1iê  c         KÄ  s%   | |  _  | |  _ t | É |  _ d  S(   N(   RC  Rÿ  Rà   RŸ  (   RI   Rÿ  RŸ  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   †
  s    		c         CÄ  s   d  S(   N(    (   RI   R_  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  •
  s    c         CÄ  sU   d j  g  |  j j É  D]" \ } } d | t | É f ^ q É } d |  j j | f S(   Ns   , s   %s=%ss   %s(%s)(   R»   RC  R	  RÊ   R¯  RK   (   RI   R  R  R”   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ®
  s    A(   RK   RL   Rq   t   quietRc   RN  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  û
  s   	t	   CGIServerc           BÄ  s   e  Z e Z d  Ñ  Z RS(   c         Ä  s3   d d l  m } á  f d Ü  } | É  j | É d  S(   Niˇˇˇˇ(   t
   CGIHandlerc         Ä  s   |  j  d d É à  |  | É S(   NR€   Rï   (   R¨   (   RÁ   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyt   fixed_environ±
  s    (   t   wsgiref.handlersR  RN  (   RI   R_  R  R  (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN  Ø
  s    (   RK   RL   R©   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  ≠
  s   t   FlupFCGIServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  sN   d d  l  } |  j j d |  j |  j f É | j j j | |  j ç j É  d  S(   Niˇˇˇˇt   bindAddress(	   t   flup.server.fcgiRC  R¨   Rÿ  RŸ  t   servert   fcgit
   WSGIServerRN  (   RI   R_  t   flup(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ∏
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR
  ∑
  s   t   WSGIRefServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         Ä  s   d d l  m â  m } d d l  m } d d  l â d à  f á  á f d Ü  É  Y} à j j d | É } à j j d | É } d à j k rƒ t | d	 É à j	 k rƒ d
 | f á f d Ü  É  Y} qƒ n  | à j à j
 | | | É } | j É  d  S(   Niˇˇˇˇ(   t   WSGIRequestHandlerR  (   t   make_servert   FixedHandlerc           Ä  s#   e  Z d  Ñ  Z á  á f d Ü  Z RS(   c         SÄ  s   |  j  d S(   Ni    (   t   client_address(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   address_string≈
  s    c          Ä  s   à j  s à  j |  | é  Sd  S(   N(   R  t   log_request(   R”   t   kw(   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  «
  s    	(   RK   RL   R  R  (    (   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  ƒ
  s   	t   handler_classt   server_classR”  t   address_familyt
   server_clsc           Ä  s   e  Z à  j Z RS(    (   RK   RL   t   AF_INET6R  (    (   t   socket(    s&   /home/lgardner/git/professor/bottle.pyR  –
  s   (   t   wsgiref.simple_serverR  R  R  R  RC  Rœ   Rÿ  Rh   t   AF_INETRŸ  t   serve_forever(   RI   RÔ   R  R  R  t   handler_clsR  t   srv(    (   R  RI   R  s&   /home/lgardner/git/professor/bottle.pyRN  ø
  s    "(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  æ
  s   t   CherryPyServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s÷   d d l  m } |  j |  j f |  j d <| |  j d <|  j j d É } | r[ |  j d =n  |  j j d É } | rÄ |  j d =n  | j |  j ç  } | r§ | | _ n  | r∂ | | _ n  z | j	 É  Wd  | j
 É  Xd  S(   Niˇˇˇˇ(   t
   wsgiservert	   bind_addrt   wsgi_appt   certfilet   keyfile(   t   cherrypyR%  Rÿ  RŸ  RC  Rœ   t   CherryPyWSGIServert   ssl_certificatet   ssl_private_keyRò   t   stop(   RI   R_  R%  R(  R)  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  ÿ
  s"    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR$  ◊
  s   t   WaitressServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s0   d d l  m } | | d |  j d |  j Éd  S(   Niˇˇˇˇ(   t   serveRÿ  RŸ  (   t   waitressR0  Rÿ  RŸ  (   RI   R_  R0  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ò
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR/  
  s   t   PasteServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  se   d d l  m } d d l m } | | d |  j É} | j | d |  j d t |  j É |  j	 çd  S(   Niˇˇˇˇ(   t
   httpserver(   t   TransLoggert   setup_console_handlerRÿ  RŸ  (
   t   pasteR3  t   paste.transloggerR4  R  R0  Rÿ  Rá   RŸ  RC  (   RI   R_  R3  R4  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  ˜
  s
    !(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR2  ˆ
  s   t   MeinheldServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s:   d d l  m } | j |  j |  j f É | j | É d  S(   Niˇˇˇˇ(   R  (   t   meinheldR  t   listenRÿ  RŸ  RN  (   RI   R_  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN     s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  ˇ
  s   t   FapwsServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   sA    Extremely fast webserver using libev. See http://www.fapws.org/ c         Ä  s÷   d d  l  j } d d l m } m } |  j } t | j d É d k rV t | É } n  | j	 |  j
 | É d t j k rô |  j rô t d É t d É n  | j | É á  f d Ü  } | j d	 | f É | j É  d  S(
   Niˇˇˇˇ(   Rå  Rı   i˛ˇˇˇgöôôôôôŸ?t   BOTTLE_CHILDs3   WARNING: Auto-reloading does not work with Fapws3.
s/            (Fapws3 breaks python thread support)
c         Ä  s   t  |  d <à  |  | É S(   Ns   wsgi.multiprocess(   Rq   (   RÁ   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRÔ     s    
Rï   (   t   fapws._evwsgit   _evwsgit   fapwsRå  Rı   RŸ  Râ   t   SERVER_IDENTRá   Rò   Rÿ  Rè  RÁ   R  t   _stderrt   set_base_modulet   wsgi_cbRN  (   RI   R_  t   evwsgiRå  Rı   RŸ  RÔ   (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN    s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR;    s   t   TornadoServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s<    The super hyped asynchronous server by facebook. Untested. c         CÄ  s~   d d  l  } d d  l } d d  l } | j j | É } | j j | É } | j d |  j d |  j	 É | j
 j j É  j É  d  S(   NiˇˇˇˇRŸ  t   address(   t   tornado.wsgit   tornado.httpservert   tornado.ioloopRÇ  t   WSGIContainerR3  t
   HTTPServerR:  RŸ  Rÿ  t   ioloopt   IOLoopt   instanceRò   (   RI   R_  t   tornadot	   containerR  (    (    s&   /home/lgardner/git/professor/bottle.pyRN    s
    $(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRE    s   t   AppEngineServerc           BÄ  s   e  Z d  Z e Z d Ñ  Z RS(   s     Adapter for Google App Engine. c         Ä  sa   d d l  m â t j j d É } | rP t | d É rP á  á f d Ü  | _ n  à j à  É d  S(   Niˇˇˇˇ(   t   utilR   t   mainc           Ä  s   à j  à  É S(   N(   t   run_wsgi_app(    (   R_  RR  (    s&   /home/lgardner/git/professor/bottle.pyR!   /  s    (   t   google.appengine.ext.webappRR  R   RF  Rœ   R2   RS  RT  (   RI   R_  RI  (    (   R_  RR  s&   /home/lgardner/git/professor/bottle.pyRN  )  s
    (   RK   RL   Rp   R©   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRQ  &  s   t   TwistedServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  sß   d d l  m } m } d d l m } d d l m } | É  } | j É  | j d d | j	 É | j
 | j | | | É É } | j |  j | d |  j É| j É  d  S(   Niˇˇˇˇ(   R  RÇ  (   t
   ThreadPool(   t   reactort   aftert   shutdownt	   interface(   t   twisted.webR  RÇ  t   twisted.python.threadpoolRW  t   twisted.internetRX  Rò   t   addSystemEventTriggerR.  t   Sitet   WSGIResourcet	   listenTCPRŸ  Rÿ  RN  (   RI   R_  R  RÇ  RW  RX  t   thread_poolt   factory(    (    s&   /home/lgardner/git/professor/bottle.pyRN  5  s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRV  3  s   t   DieselServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  s3   d d l  m } | | d |  j É} | j É  d  S(   Niˇˇˇˇ(   t   WSGIApplicationRŸ  (   t   diesel.protocols.wsgiRf  RŸ  RN  (   RI   R_  Rf  RÔ   (    (    s&   /home/lgardner/git/professor/bottle.pyRN  C  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRe  A  s   t   GeventServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s„    Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    c         Ä  sı   d d l  m } m } m } t t j É  | j É sI d } t | É Ç n  |  j j d d  É sg | } n  |  j
 rv d  n d |  j d <|  j |  j f } | j | | |  j ç â  d t j k rÁ d d  l } | j | j á  f d Ü  É n  à  j É  d  S(	   Niˇˇˇˇ(   RÇ  t   pywsgiR5  s9   Bottle requires gevent.monkey.patch_all() (before import)t   fastR
   t   logR<  c         Ä  s
   à  j  É  S(   N(   R.  (   R0   Rÿ   (   R  (    s&   /home/lgardner/git/professor/bottle.pyR!   [  s    (   R   RÇ  Ri  R5  R>   R4  RÑ  RC  R—   Rg   R  Rÿ  RŸ  R  Rè  RÁ   t   signalt   SIGINTR!  (   RI   R_  RÇ  Ri  R5  R¡   RF  Rl  (    (   R  s&   /home/lgardner/git/professor/bottle.pyRN  P  s     	(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRh  I  s   t   GeventSocketIOServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  sB   d d l  m } |  j |  j f } | j | | |  j ç j É  d  S(   Niˇˇˇˇ(   R  (   t   socketioR  Rÿ  RŸ  t   SocketIOServerRC  R!  (   RI   R_  R  RF  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  `  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRn  _  s   t   GunicornServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s?    Untested. See http://gunicorn.org/configure.html for options. c         Ä  ss   d d l  m } i d |  j t |  j É f d 6â  à  j |  j É d | f á  á f d Ü  É  Y} | É  j É  d  S(   Niˇˇˇˇ(   t   Applications   %s:%dRg  t   GunicornApplicationc           Ä  s&   e  Z á  f d  Ü  Z á f d Ü  Z RS(   c         Ä  s   à  S(   N(    (   RI   t   parsert   optsR”   (   Rı   (    s&   /home/lgardner/git/professor/bottle.pyt   inito  s    c         Ä  s   à  S(   N(    (   RI   (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRW  r  s    (   RK   RL   Rv  RW  (    (   Rı   R_  (    s&   /home/lgardner/git/professor/bottle.pyRs  n  s   (   t   gunicorn.app.baseRr  Rÿ  Rà   RŸ  RJ  RC  RN  (   RI   R_  Rr  Rs  (    (   Rı   R_  s&   /home/lgardner/git/professor/bottle.pyRN  h  s
    #(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRq  f  s   t   EventletServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s
    Untested c         CÄ  sÄ   d d l  m } m } y0 | j | |  j |  j f É | d |  j ÉWn3 t k
 r{ | j | |  j |  j f É | É n Xd  S(   Niˇˇˇˇ(   RÇ  R:  t
   log_output(   t   eventletRÇ  R:  R  Rÿ  RŸ  R  RJ  (   RI   R_  RÇ  R:  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  z  s    !(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx  x  s   t   RocketServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  sC   d d l  m } | |  j |  j f d i | d 6É } | j É  d  S(   Niˇˇˇˇ(   t   RocketRÇ  R'  (   t   rocketR|  Rÿ  RŸ  Rò   (   RI   R_  R|  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ü  s    %(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{  Ñ  s   t   BjoernServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s?    Fast server written in C: https://github.com/jonashaag/bjoern c         CÄ  s*   d d l  m } | | |  j |  j É d  S(   Niˇˇˇˇ(   RN  (   t   bjoernRN  Rÿ  RŸ  (   RI   R_  RN  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  é  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR~  å  s   t
   AutoServerc           BÄ  s,   e  Z d  Z e e e e e g Z d Ñ  Z	 RS(   s    Untested. c         CÄ  sR   xK |  j  D]@ } y& | |  j |  j |  j ç j | É SWq
 t k
 rI q
 Xq
 Wd  S(   N(   t   adaptersRÿ  RŸ  RC  RN  R   (   RI   R_  t   sa(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ñ  s
    &(
   RK   RL   Rp   R/  R2  RV  R$  R  RÅ  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÄ  ì  s   Rƒ  R  R1  R*  R6  t   fapws3RO  t   gaet   twistedt   dieselR9  t   gunicornRz  t   geventSocketIOR}  R  c         KÄ  s∏   d |  k r |  j  d d É n	 |  d f \ } }  | t j k rL t | É n  |  s] t j | S|  j É  r} t t j | |  É S| j  d É d } t j | | | <t d | |  f | É S(   sˇ   Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    R”  i   RL  i    s   %s.%sN(   R@  Rg   R   RF  RQ  t   isalnumRh   t   eval(   Rµ   Rm  RI  t   package_name(    (    s&   /home/lgardner/git/professor/bottle.pyRW  Ω  s    0   c         CÄ  sX   t  t a } z0 t j É  } t |  É } t | É r8 | S| SWd t j | É | a Xd S(   sﬁ    Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. N(   R©   t   NORUNt   default_appRÇ  RW  RI  R+  (   Rµ   t   nr_oldRπ  R=  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_app—  s    s	   127.0.0.1iê  c	         KÄ  sÿ  t  r
 d S| r~t j j d É r~z1yd }
 t j d d d d É \ } }
 t j | É x· t j j	 |
 É r=t
 j g t
 j } t j j É  } d | d <|
 | d <t j | d	 | É} x3 | j É  d k rÔ t j |
 d É t j | É qΩ W| j É  d
 k r] t j j	 |
 É r$t j |
 É n  t
 j | j É  É q] q] WWn t k
 rRn XWd t j j	 |
 É ryt j |
 É n  Xd Sy·| d k	 röt | É n  |  p¶t É  }  t |  t É r«t |  É }  n  t |  É sÊt d |  É Ç n  x! | pÚg  D] } |  j | É qÛW| t k r(t j | É } n  t | t É rFt  | É } n  t | t! É rp| d | d | |	 ç } n  t | t" É sít d | É Ç n  | j# pû| | _# | j# sÓt$ d t% t& | É f É t$ d | j' | j( f É t$ d É n  | rQt j j d É }
 t) |
 | É } | è | j* |  É Wd QX| j+ d k r^t
 j d
 É q^n | j* |  É Wnr t k
 rrnb t, t- f k
 rãÇ  nI | söÇ  n  t. | d | É s∂t/ É  n  t j | É t
 j d
 É n Xd S(   sº   Start a server instance. This method blocks until the server terminates.

        :param app: WSGI application or target string supported by
               :func:`load_app`. (default: :func:`default_app`)
        :param server: Server adapter to use. See :data:`server_names` keys
               for valid names or pass a :class:`ServerAdapter` subclass.
               (default: `wsgiref`)
        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on
               all interfaces including the external one. (default: 127.0.0.1)
        :param port: Server port to bind to. Values below 1024 require root
               privileges. (default: 8080)
        :param reloader: Start auto-reloading server? (default: False)
        :param interval: Auto-reloader interval in seconds (default: 1)
        :param quiet: Suppress output to stdout and stderr? (default: False)
        :param options: Options passed to the server adapter.
     NR<  Rù   s   bottle.t   suffixs   .lockt   truet   BOTTLE_LOCKFILER◊  i   s   Application is not callable: %rRÿ  RŸ  s!   Unknown or unsupported server: %rs,   Bottle v%s server starting up (using %s)...
s   Listening on http://%s:%d/
s   Hit Ctrl-C to quit.

t   reloadR  (0   Rå  Rè  RÁ   Rœ   Rg   t   tempfilet   mkstempRJ   Rä   Rï  R   t
   executablet   argvRÒ  t
   subprocesst   Popent   pollt   utimeR+  t   sleept   unlinkt   exitRj  t   _debugRç  R>   R?  Rè  RI  R£   R   t   server_namesRW  R˛   R  R  RA  t   __version__RÊ   Rÿ  RŸ  t   FileCheckerThreadRN  R1  Rk  Rl  Rh   R   (   RÔ   R  Rÿ  RŸ  t   intervalt   reloaderR  RÒ   RÃ  RS  t   lockfilet   fdR”   RÁ   RÇ   R  t   bgcheck(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ﬂ  sà      

  	 
R¢  c           BÄ  s2   e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   sw    Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. c         CÄ  s0   t  j j |  É | | |  _ |  _ d  |  _ d  S(   N(   R4  t   ThreadRc   R•  R£  Rg   R1  (   RI   R•  R£  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ?  s    c         CÄ  s[  t  j j } d Ñ  } t É  } xq t t j j É  É D]Z } t | d d É } | d d k ri | d  } n  | r4 | | É r4 | | É | | <q4 q4 Wx¬ |  j	 sV| |  j
 É s‘ | |  j
 É t j É  |  j d k  rÍ d	 |  _	 t j É  n  xV t | j É  É D]B \ } } | | É s(| | É | k r˝ d
 |  _	 t j É  Pq˝ q˝ Wt j |  j É qï Wd  S(   Nc         SÄ  s   t  j |  É j S(   N(   Rè  Rø  R¡  (   Rä   (    (    s&   /home/lgardner/git/professor/bottle.pyR!   G  s    RA  Rï   i¸ˇˇˇs   .pyos   .pyciˇˇˇˇi   RØ   Rì  (   s   .pyos   .pyc(   Rè  Rä   Rï  R]   R[   R   RF  Râ  Rh   R1  R•  R+  R£  t   threadt   interrupt_mainR	  Rú  (   RI   Rï  t   mtimeRò  RI  Rä   t   lmtime(    (    s&   /home/lgardner/git/professor/bottle.pyRN  E  s(    		  &		
c         CÄ  s   |  j  É  d  S(   N(   Rò   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   __enter__[  s    c         CÄ  s8   |  j  s d |  _  n  |  j É  | d  k	 o7 t | t É S(   NRû  (   R1  R»   Rg   R  Rj  (   RI   t   exc_typet   exc_valt   exc_tb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __exit__^  s    	 
(   RK   RL   Rp   Rc   RN  R≠  R±  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR¢  ;  s
   			t   TemplateErrorc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s   t  j |  d | É d  S(   NiÙ  (   R§   Rc   (   RI   RW   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   m  s    (   RK   RL   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR≤  l  s   t   BaseTemplatec           BÄ  st   e  Z d  Z d d d d g Z i  Z i  Z d d g  d d Ñ Z e g  d Ñ É Z	 e d Ñ  É Z
 d	 Ñ  Z d
 Ñ  Z RS(   s2    Base class and minimal API for template adapters t   tplt   htmlt   thtmlt   stplR=   c         KÄ  s+  | |  _  t | d É r$ | j É  n | |  _ t | d É rE | j n d |  _ g  | D] } t j j | É ^ qU |  _	 | |  _
 |  j j É  |  _ |  j j | É |  j rÙ |  j  rÙ |  j |  j  |  j	 É |  _ |  j sÙ t d t | É É Ç qÙ n  |  j r|  j rt d É Ç n  |  j |  j ç  d S(   s=   Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        Ro  R∆  s   Template %s not found.s   No template specified.N(   Rì   R2   Ro  Rz  R∆  Rg   Rè  Rä   Rê  Rû  R(   t   settingsRÒ  RJ  Rô  R≤  RÊ   R˘   (   RI   Rz  Rì   Rû  R(   R∏  R    (    (    s&   /home/lgardner/git/professor/bottle.pyRc   w  s    	$!(		c         CÄ  s  | s t  d É d g } n  t j j | É rZ t j j | É rZ t  d É t j j | É Sx± | D]© } t j j | É t j } t j j t j j | | É É } | j | É s∂ qa n  t j j | É rÃ | Sx; |  j	 D]0 } t j j d | | f É r÷ d | | f Sq÷ Wqa Wd S(   s{    Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. s2   The template lookup path list should not be empty.RL  s,   Absolute template path names are deprecated.s   %s.%sN(
   RY   Rè  Rä   t   isabsRú  Rê  Rí  R»   R¬  t
   extensions(   Rj   Rì   Rû  t   spathR°  t   ext(    (    s&   /home/lgardner/git/professor/bottle.pyRô  ë  s     
$
!  c         GÄ  s;   | r, |  j  j É  |  _  | d |  j  | <n |  j  | Sd S(   sB    This reads or sets the global settings stored in class.settings. i    N(   R∏  RÒ  (   Rj   Ra   R”   (    (    s&   /home/lgardner/git/professor/bottle.pyt   global_config¶  s    c         KÄ  s
   t  Ç d S(   sô    Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        N(   t   NotImplementedError(   RI   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   Ø  s    c         OÄ  s
   t  Ç d S(   sF   Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        N(   Ræ  (   RI   R”   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   render∂  s    N(   RK   RL   Rp   R∫  R∏  t   defaultsRg   Rc   t   classmethodRô  RΩ  R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR≥  q  s   		t   MakoTemplatec           BÄ  s   e  Z d  Ñ  Z d Ñ  Z RS(   c         KÄ  s¥   d d l  m } d d l m } | j i |  j d 6É | j d t t É É | d |  j	 | ç } |  j
 râ | |  j
 d | | ç|  _ n' | d |  j d	 |  j d | | ç |  _ d  S(
   Niˇˇˇˇ(   t   Template(   t   TemplateLookupR`  t   format_exceptionst   directoriesRû  t   uriR∆  (   t   mako.templateR√  t   mako.lookupRƒ  RJ  R(   R¨   R  R±   Rû  Rz  R¥  Rì   R∆  (   RI   RC  R√  Rƒ  Rû  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   ¡  s    	c         OÄ  sJ   x | D] } | j  | É q W|  j j É  } | j  | É |  j j | ç  S(   N(   RJ  R¿  RÒ  R¥  Rø  (   RI   R”   R.  t   dictargt	   _defaults(    (    s&   /home/lgardner/git/professor/bottle.pyRø  Ã  s
     (   RK   RL   R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR¬  ¿  s   	t   CheetahTemplatec           BÄ  s   e  Z d  Ñ  Z d Ñ  Z RS(   c         KÄ  s~   d d l  m } t j É  |  _ i  |  j _ |  j j g | d <|  j rb | d |  j | ç |  _ n | d |  j | ç |  _ d  S(   Niˇˇˇˇ(   R√  t
   searchListRz  R«  (	   t   Cheetah.TemplateR√  R4  R5  R  t   varsRz  R¥  R∆  (   RI   RC  R√  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   ‘  s    	c         OÄ  sj   x | D] } | j  | É q W|  j j j  |  j É |  j j j  | É t |  j É } |  j j j É  | S(   N(   RJ  R  Rœ  R¿  Rá   R¥  R~  (   RI   R”   R.  R   Rx  (    (    s&   /home/lgardner/git/professor/bottle.pyRø  ﬁ  s     (   RK   RL   R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÃ  ”  s   	
t   Jinja2Templatec           BÄ  s,   e  Z d d i  d  Ñ Z d Ñ  Z d Ñ  Z RS(   c         KÄ  s„   d d l  m } m } d | k r1 t d É Ç n  | d | |  j É | ç |  _ | rk |  j j j | É n  | rá |  j j j | É n  | r£ |  j j	 j | É n  |  j
 r« |  j j |  j
 É |  _ n |  j j |  j É |  _ d  S(   Niˇˇˇˇ(   t   Environmentt   FunctionLoaderRù   ss   The keyword argument `prefix` has been removed. Use the full jinja2 environment name line_statement_prefix instead.t   loader(   t   jinja2R—  R“  RÑ  R”  R◊  Rí   RJ  t   testst   globalsRz  t   from_stringR¥  t   get_templateR∆  (   RI   Rí   R’  R÷  R.  R—  R“  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   Ë  s       	c         OÄ  sJ   x | D] } | j  | É q W|  j j É  } | j  | É |  j j | ç  S(   N(   RJ  R¿  RÒ  R¥  Rø  (   RI   R”   R.  R   RÀ  (    (    s&   /home/lgardner/git/professor/bottle.pyRø  ˆ  s
     c         CÄ  sQ   |  j  | |  j É } | s d  St | d É è } | j É  j |  j É SWd  QXd  S(   NRπ  (   Rô  Rû  Rä  Ro  RE   R(   (   RI   Rì   R°  Rÿ   (    (    s&   /home/lgardner/git/professor/bottle.pyR”  ¸  s
     N(   RK   RL   Rg   R˘   Rø  R”  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR–  Á  s   	t   SimpleTemplatec           BÄ  sb   e  Z e e d d  Ñ Z e d Ñ  É Z e d Ñ  É Z d d Ñ Z	 d d Ñ Z
 d Ñ  Z d Ñ  Z RS(   c         Ä  sh   i  |  _  |  j â  á  f d Ü  |  _ á  á f d Ü  |  _ | |  _ | rd |  j |  j |  _ |  _ n  d  S(   Nc         Ä  s   t  |  à  É S(   N(   R/   (   R    (   RB   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    c         Ä  s   à t  |  à  É É S(   N(   R/   (   R    (   RB   t   escape_func(    s&   /home/lgardner/git/professor/bottle.pyR!   	  s    (   Ré  R(   t   _strt   _escapet   syntax(   RI   R⁄  t   noescapeR›  RR   (    (   RB   R⁄  s&   /home/lgardner/git/professor/bottle.pyR˘     s    			c         CÄ  s   t  |  j |  j p d d É S(   Ns   <string>R<   (   RÆ   R`  R∆  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   co  s    c         CÄ  sª   |  j  } | s9 t |  j d É è } | j É  } Wd  QXn  y t | É d } } Wn1 t k
 rÉ t d É t | d É d } } n Xt | d | d |  j É} | j	 É  } | j
 |  _
 | S(   NRπ  R=   s;   Template encodings other than utf8 are no longer supported.R)   R(   R›  (   Rz  Rä  R∆  Ro  R/   Rf  RY   t
   StplParserR›  t	   translateR(   (   RI   Rz  Rÿ   R(   Rt  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR`    s    	
c         KÄ  s0   | d  k r t d t É n  | | f | d <d  S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?t   _rebase(   Rg   RY   R©   (   RI   t   _envR‘   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyR‚  "  s    
c         KÄ  sÑ   | d  k r t d t É n  | j É  } | j | É | |  j k ri |  j d | d |  j É |  j | <n  |  j | j | d | É S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?Rì   Rû  t   _stdout(	   Rg   RY   R©   RÒ  RJ  Ré  R¯  Rû  t   execute(   RI   R„  R‘   R.  R◊  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _include(  s    
%c         CÄ  s  |  j  j É  } | j | É | j i
 | d 6| j d 6t j |  j | É d 6t j |  j | É d 6d  d 6|  j	 d 6|  j
 d 6| j d 6| j d	 6| j d
 6É t |  j | É | j d É r˝ | j d É \ } } d j | É | d <| 2|  j | | | ç S| S(   NR‰  t
   _printlistt   includet   rebaseR‚  R€  R‹  Rœ   R¨   t   definedRï   Rå  (   R¿  RÒ  RJ  t   extendRM   R  RÊ  R‚  Rg   R€  R‹  Rœ   R¨   R  Rä  Rﬂ  R—   R»   (   RI   R‰  R.  R◊  t   subtplt   rargs(    (    s&   /home/lgardner/git/professor/bottle.pyRÂ  2  s    c         OÄ  sT   i  } g  } x | D] } | j  | É q W| j  | É |  j | | É d j | É S(   sA    Render the template using keyword arguments as local variables. Rï   (   RJ  RÂ  R»   (   RI   R”   R.  R◊  R   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRø  B  s      N(   RK   RL   RÄ  Rq   Rg   R˘   Rr   Rﬂ  R`  R‚  RÊ  RÂ  Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRŸ    s   	
	t   StplSyntaxErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÓ  K  s    R‡  c           BÄ  sÒ   e  Z d  Z i  Z d Z e j d d É Z e d 7Z e d 7Z e d 7Z e d 7Z e d 7Z e d	 7Z e d
 7Z d Z d e Z d Z d d d Ñ Z
 d Ñ  Z d Ñ  Z e e e É Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z RS(   s    Parser for stpl templates. sà   ((?m)[urbURB]?(?:''(?!')|""(?!")|'{6}|"{6}|'(?:[^\\']|\\.)+?'|"(?:[^\\"]|\\.)+?"|'{3}(?:[^\\]|\\.|\n)+?'{3}|"{3}(?:[^\\]|\\.|\n)+?"{3}))s   |\nRï   s   |(#.*)s   |([\[\{\(])s   |([\]\}\)])sW   |^([ \t]*(?:if|for|while|with|try|def|class)\b)|^([ \t]*(?:elif|else|except|finally)\b)s?   |((?:^|;)[ \t]*end[ \t]*(?=(?:%(block_close)s[ \t]*)?\r?$|;|#))s   |(%(block_close)s[ \t]*(?=$))s   |(\r?\n)s8   (?m)^[ 	]*(\\?)((%(line_start)s)|(%(block_start)s))(%%?)s2   %%(inline_start)s((?:%s|[^'"
]*?)+)%%(inline_end)ss   <% %> % {{ }}R=   c         CÄ  sv   t  | | É | |  _ |  _ |  j | p. |  j É g  g  |  _ |  _ d \ |  _ |  _ d \ |  _	 |  _
 d |  _ d  S(   Ni   i    (   i   i    (   i    i    (   R/   Rz  R(   t
   set_syntaxt   default_syntaxt   code_buffert   text_buffert   linenoRú   t   indentt
   indent_modt   paren_depth(   RI   Rz  R›  R(   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   n  s    c         CÄ  s   |  j  S(   s=    Tokens as a space separated string (default: <% %> % {{ }}) (   t   _syntax(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_syntaxv  s    c         CÄ  sŒ   | |  _  | j É  |  _ | |  j k r´ d } t t j |  j É } t t | j É  | É É } |  j	 |  j
 |  j f } g  | D] } t j | | É ^ q| } | |  j | <n  |  j | \ |  _ |  _ |  _ d  S(   Ns:   block_start block_close line_start inline_start inline_end(   R˜  R@  t   _tokenst	   _re_cachet   mapRÄ   R´   R]   R‰  t	   _re_splitt   _re_tokt   _re_inlRÆ   t   re_splitt   re_tokt   re_inl(   RI   R›  Rd  t   etokenst   pattern_varst   patternsRÇ   (    (    s&   /home/lgardner/git/professor/bottle.pyRÔ  z  s    	&c         CÄ  sÓ  |  j  r t d É Ç n  xüt rπ|  j j |  j |  j  É } | rµ|  j |  j  |  j  | j É  !} |  j j | É |  j  | j	 É  7_  | j
 d É r
|  j |  j  j d É \ } } } |  j j | j
 d É | j
 d É | | É |  j  t | | É d 7_  q n | j
 d É rât d É |  j |  j  j d É \ } } } |  j j | j
 d É | | É |  j  t | | É d 7_  q n  |  j É  |  j d t | j
 d É É É q Pq W|  j j |  j |  j  É |  j É  d	 j |  j É S(
   Ns   Parser is a one time instance.i   s   
i   i   s#   Escape code lines with a backslash.t	   multilinei   Rï   (   Rú   RÑ  R©   Rˇ  Rô  Rz  Rò   RÚ  R   Rö   R~   R®  R}   RY   t
   flush_textt	   read_codeR  R»   RÒ  (   RI   R   R∞  t   lineRí  Rƒ   (    (    s&   /home/lgardner/git/professor/bottle.pyR·  à  s2    	 	 ".
"!
"
c      	   CÄ  su  d \ } } xbt  rp|  j j |  j |  j É } | sw | |  j |  j 7} t |  j É |  _ |  j | j É  | É d  S| |  j |  j |  j | j É  !7} |  j | j	 É  7_ | j
 É  \	 } } } } }	 }
 } } } | sÏ |  j d k r|	 s¯ |
 r| |	 p|
 7} q n  | r!| | 7} q | r[| } | rm| j É  j |  j d É rmt } qmq | r}|  j d 7_ | | 7} q | r±|  j d k r§|  j d 8_ n  | | 7} q |	 rŸ|	 d } |  _ |  j d 7_ q |
 rÚ|
 d } |  _ q | r
|  j d 8_ q | r,| rt } qm| | 7} q |  j | j É  | É |  j d 7_ d \ } } |  _ | s Pq q Wd  S(   NRï   i    i   iˇˇˇˇ(   Rï   Rï   (   Rï   Rï   i    (   R©   R   Rô  Rz  Rú   R}   t
   write_codeRP  Rò   Rö   Rô   Rˆ  RB  R˘  Rq   Rı  RÙ  RÛ  (   RI   R  t	   code_linet   commentR   R€  t   _comt   _pot   _pct   _blk1t   _blk2t   _endt   _cendt   _nl(    (    s&   /home/lgardner/git/professor/bottle.pyR  ¢  sV    	$'!" 	c   	      CÄ  s‘  d j  |  j É } |  j 2| s# d  Sg  d d d |  j } } } x≤ |  j j | É D]û } | | | j É  !| j É  } } | r¨ | j | j  t t	 | j
 t É É É É n  | j d É rŒ | d c | 7<n  | j |  j | j d É j É  É É qU W| t | É k  rî| | } | j
 t É } | d j d É rJ| d d	  | d <n( | d j d
 É rr| d d  | d <n  | j | j  t t	 | É É É n  d d j  | É } |  j | j d É d 7_ |  j | É d  S(   NRï   i    s   \
s     s   
iˇˇˇˇi   s   \\
i˝ˇˇˇs   \\
i¸ˇˇˇs   _printlist((%s,))s   , (   R»   RÚ  RÙ  R  Ró   Rò   Rö   R   R˚  RÊ   t
   splitlinesR©   RB  t   process_inlineR~   RP  R}   RÛ  t   countR	  (	   RI   R∞  t   partst   post   nlR   Rù   t   linesR`  (    (    s&   /home/lgardner/git/professor/bottle.pyR  —  s.      + )
  "c         CÄ  s$   | d d k r d | d Sd | S(   Ni    RÊ  s   _str(%s)i   s   _escape(%s)(    (   RI   t   chunk(    (    s&   /home/lgardner/git/professor/bottle.pyR  Ê  s     c         CÄ  sX   |  j  | | É \ } } d |  j |  j } | | j É  | d 7} |  j j | É d  S(   Ns     s   
(   t   fix_backward_compatibilityRÙ  Rı  RQ  RÒ  R   (   RI   R  R  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  Í  s    c         CÄ  s7  | j  É  j d  d É } | rë | d d k rë t d É t | É d k rT d | f St | É d k rz d t | É | f Sd	 t | É | f Sn  |  j d k r-| j  É  r-d
 | k r-t j d | É } | r-t d É | j	 d É } |  j
 j |  j É j | É |  _
 | |  _ | | j d
 d É f Sn  | | f S(   Ni   i    RË  RÈ  s2   The include and rebase keywords are functions now.i   s   _printlist([base])s   _=%s(%r)s   _=%s(%r, %s)t   codings   #.*coding[:=]\s*([-\w.]+)s4   PEP263 encoding strings in templates are deprecated.s   coding*(   s   includes   rebase(   RP  R@  Rg   RY   R}   RZ   RÛ  RÄ   Rû   R~   Rz  R@   R(   RE   R   (   RI   R  R  R  R   RB   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s"    
 
 (
!	N(   RK   RL   Rp   R˙  R˝  R   R˛  R¸  R  Rg   Rc   R¯  RÔ  R  R›  R·  R  R  R  R	  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR‡  N  s0   







				/		c          OÄ  se  |  r |  d n d } | j d t É } | j d t É } t | É | f } | t k s^ t r| j d i  É } t | | É r¶ | t | <| rt | j | ç  qqd | k s÷ d | k s÷ d | k s÷ d | k rı | d	 | d
 | | ç t | <q| d | d
 | | ç t | <n  t | s2t	 d d | É n  x |  d D] } | j
 | É q=Wt | j | É S(   sÍ   
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    i    t   template_adaptert   template_lookupt   template_settingss   
t   {t   %t   $Rz  Rû  Rì   iÙ  s   Template (%s) not foundi   N(   Rg   R—   RŸ  t   TEMPLATE_PATHt   idt	   TEMPLATESR±   R>   R˘   R±  RJ  Rø  (   R”   R.  R¥  t   adapterRû  t   tplidR∏  R   (    (    s&   /home/lgardner/git/professor/bottle.pyRb    s$    
 0
 R  c         Ä  s   á  á f d Ü  } | S(   s…   Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    c         Ä  s(   t  j à  É á á  á f d Ü  É } | S(   Nc          Ä  sg   à |  | é  } t  | t t f É rJ à  j É  } | j | É t à | ç S| d  k rc t à à  É S| S(   N(   R>   R]   R9   RÒ  RJ  Rb  Rg   (   R”   R.  t   resultt   tplvars(   R¿  Rf   t   tpl_name(    s&   /home/lgardner/git/professor/bottle.pyRP   +  s    (   RM   R  (   Rf   RP   (   R¿  R+  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  *  s    $
(    (   R+  R¿  R0  (    (   R¿  R+  s&   /home/lgardner/git/professor/bottle.pyR?     s    
s   ./s   ./views/s	   ../views/s   I'm a teapoti¢  s   Unprocessable Entityi¶  s   Precondition Requiredi¨  s   Too Many Requestsi≠  s   Request Header Fields Too LargeiØ  s   Network Authentication Requirediˇ  c         cÄ  s+   |  ]! \ } } | d  | | f f Vq d S(   s   %d %sN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>S  s    sÇ  
%%try:
    %%from %s import DEBUG, HTTP_CODES, request, touni
    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
    <html>
        <head>
            <title>Error: {{e.status}}</title>
            <style type="text/css">
              html {background-color: #eee; font-family: sans;}
              body {background-color: #fff; border: 1px solid #ddd;
                    padding: 15px; margin: 15px;}
              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}
            </style>
        </head>
        <body>
            <h1>Error: {{e.status}}</h1>
            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>
               caused an error:</p>
            <pre>{{e.body}}</pre>
            %%if DEBUG and e.exception:
              <h2>Exception:</h2>
              <pre>{{repr(e.exception)}}</pre>
            %%end
            %%if DEBUG and e.traceback:
              <h2>Traceback:</h2>
              <pre>{{e.traceback}}</pre>
            %%end
        </body>
    </html>
%%except ImportError:
    <b>ImportError:</b> Could not generate the error page. Please add bottle to
    the import path.
%%end
s
   bottle.exts   .exts	   bottle_%ss
   Bottle %s
s"   
Error: No application specified.
RL  Rv  t	   localhostR”  t   ]s   []Rÿ  RŸ  R  R§  RÒ   RÃ  (  Rp   t
   __future__R    t
   __author__R°  t   __license__RK   t   optparseR   t   _cmd_parsert
   add_optiont   _optt
   parse_argst   _cmd_optionst	   _cmd_argsR  R¬  t   gevent.monkeyR   t   monkeyt	   patch_allR÷  Rƒ  t   email.utilsRŒ  RM   RÈ  RG  R:  RΩ  Rè  RÄ   Rò  R   Rî  R4  R+  RT   R   R   R)  R   R   R;  R   R   t   inspectR   t   unicodedataR   t
   simplejsonR   R   R   R.   R   R†  t   django.utils.simplejsont   version_infot   pyR  t   py25R√  R   R   R   R"   R‰  RA  R†  t   http.clientt   clientt   httplibt   _threadR©  t   urllib.parseR#   R$   R÷  R%   R&   R‘  R'   Rﬁ  R  t   http.cookiesR*   t   collectionsR+   R9   RË  t   ioR,   t   configparserR-   Rá   R?  R?   Rù  RI  R˚  R6   R5   t   urlparset   urllibt   Cookiet   cPickleR7   R8   R¡   RU   RV   t   UserDictR:   RA   Rä  RÆ   RC   R/   R©  RG   RH   RN   Rq   RY   R^   R˚  R_   Rr   Rt   Rm  Rv   Rw   Rx   Ry   Rz   R{   RÉ   RÑ   RÌ   R  RÉ  R  R  R  Rg   R6  R7  R8  R  t   ResponseR9  R§   R<  R!  R"  R@  RU  Rä  R  RÖ  R]   RÛ   R[   RÅ  Rt  Rw  R  Rî  R±  R¥  Rµ  R   R©   RÃ  R#  R"  RÂ  R¬  Rë  RÂ  R&  Rå  RÌ  RÄ  RÛ  RX  R8  R  R  RA  Rœ   RZ  R\  R^  RØ   RE  R/  R   RL  RŸ   R  R  R
  R  R$  R/  R2  R8  R;  RE  RQ  RV  Re  Rh  Rn  Rq  Rx  R{  R~  RÄ  R†  RW  Rè  Rü  RN  R®  R¢  R≤  R≥  R¬  RÃ  R–  RŸ  RÓ  R‡  Rb  t   mako_templatet   cheetah_templatet   jinja2_templateR?  t	   mako_viewt   cheetah_viewt   jinja2_viewR$  R&  R±   Rå  t	   responsest
   HTTP_CODESR	  R  Rc  R7  Rh  R5  RÔ   Rç  RÇ  RI  Rº  t   optR”   Rt  t   versionRû  t
   print_helpRä   R)  RF  R¨   Rg  Rÿ  RŸ  t   rfindRM  RP  Rà   Rì  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyt   <module>   s  	 ¿   		.	"									»wˇ °ˇ û	‡
$I/2ÕVH
Q				
				
					
	


		Z1OH¥			





$
		
(	

*(#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

#cd $parent_path
echo $parent_path

python2.7 ./run.py -p 8081
#!/bin/python
import bottle as bottle
from bottle import *

#Static
@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static/')

#Template
@route('/')
def main():
    return template('index.tpl')

@post("/GID")
def post_gid():
    USER_IN = request.query.get("gid") or ""
    print("x")
    return template('accounts.tpl', USER_IN=USER_IN)
@get("/GID")
def get_gid():
        #print(request.query.get("User0"))
        #print(request.query.get('User1'))
        print(request.query.get('confirmed'))
run(host='localhost', port=8081, debug=True)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
"""

from __future__ import with_statement

__author__ = 'Marcel Hellkamp'
__version__ = '0.12.9'
__license__ = 'MIT'

# The gevent server adapter needs to patch some modules before they are imported
# This is why we parse the commandline parameters here but handle them later
if __name__ == '__main__':
    from optparse import OptionParser
    _cmd_parser = OptionParser(usage="usage: %prog [options] package.module:app")
    _opt = _cmd_parser.add_option
    _opt("--version", action="store_true", help="show version number.")
    _opt("-b", "--bind", metavar="ADDRESS", help="bind socket to ADDRESS.")
    _opt("-s", "--server", default='wsgiref', help="use SERVER as backend.")
    _opt("-p", "--plugin", action="append", help="install additional plugin/s.")
    _opt("--debug", action="store_true", help="start server in debug mode.")
    _opt("--reload", action="store_true", help="auto-reload on file changes.")
    _cmd_options, _cmd_args = _cmd_parser.parse_args()
    if _cmd_options.server and _cmd_options.server.startswith('gevent'):
        import gevent.monkey; gevent.monkey.patch_all()

import base64, cgi, email.utils, functools, hmac, imp, itertools, mimetypes,\
        os, re, subprocess, sys, tempfile, threading, time, warnings

from datetime import date as datedate, datetime, timedelta
from tempfile import TemporaryFile
from traceback import format_exc, print_exc
from inspect import getargspec
from unicodedata import normalize


try: from simplejson import dumps as json_dumps, loads as json_lds
except ImportError: # pragma: no cover
    try: from json import dumps as json_dumps, loads as json_lds
    except ImportError:
        try: from django.utils.simplejson import dumps as json_dumps, loads as json_lds
        except ImportError:
            def json_dumps(data):
                raise ImportError("JSON support requires Python 2.6 or simplejson.")
            json_lds = json_dumps



# We now try to fix 2.5/2.6/3.1/3.2 incompatibilities.
# It ain't pretty but it works... Sorry for the mess.

py   = sys.version_info
py3k = py >= (3, 0, 0)
py25 = py <  (2, 6, 0)
py31 = (3, 1, 0) <= py < (3, 2, 0)

# Workaround for the missing "as" keyword in py3k.
def _e(): return sys.exc_info()[1]

# Workaround for the "print is a keyword/function" Python 2/3 dilemma
# and a fallback for mod_wsgi (resticts stdout/err attribute access)
try:
    _stdout, _stderr = sys.stdout.write, sys.stderr.write
except IOError:
    _stdout = lambda x: sys.stdout.write(x)
    _stderr = lambda x: sys.stderr.write(x)

# Lots of stdlib and builtin differences.
if py3k:
    import http.client as httplib
    import _thread as thread
    from urllib.parse import urljoin, SplitResult as UrlSplitResult
    from urllib.parse import urlencode, quote as urlquote, unquote as urlunquote
    urlunquote = functools.partial(urlunquote, encoding='latin1')
    from http.cookies import SimpleCookie
    from collections import MutableMapping as DictMixin
    import pickle
    from io import BytesIO
    from configparser import ConfigParser
    basestring = str
    unicode = str
    json_loads = lambda s: json_lds(touni(s))
    callable = lambda x: hasattr(x, '__call__')
    imap = map
    def _raise(*a): raise a[0](a[1]).with_traceback(a[2])
else: # 2.x
    import httplib
    import thread
    from urlparse import urljoin, SplitResult as UrlSplitResult
    from urllib import urlencode, quote as urlquote, unquote as urlunquote
    from Cookie import SimpleCookie
    from itertools import imap
    import cPickle as pickle
    from StringIO import StringIO as BytesIO
    from ConfigParser import SafeConfigParser as ConfigParser
    if py25:
        msg  = "Python 2.5 support may be dropped in future versions of Bottle."
        warnings.warn(msg, DeprecationWarning)
        from UserDict import DictMixin
        def next(it): return it.next()
        bytes = str
    else: # 2.6, 2.7
        from collections import MutableMapping as DictMixin
    unicode = unicode
    json_loads = json_lds
    eval(compile('def _raise(*a): raise a[0], a[1], a[2]', '<py3fix>', 'exec'))

# Some helpers for string/byte handling
def tob(s, enc='utf8'):
    return s.encode(enc) if isinstance(s, unicode) else bytes(s)
def touni(s, enc='utf8', err='strict'):
    return s.decode(enc, err) if isinstance(s, bytes) else unicode(s)
tonat = touni if py3k else tob

# 3.2 fixes cgi.FieldStorage to accept bytes (which makes a lot of sense).
# 3.1 needs a workaround.
if py31:
    from io import TextIOWrapper
    class NCTextIOWrapper(TextIOWrapper):
        def close(self): pass # Keep wrapped buffer open.


# A bug in functools causes it to break if the wrapper is an instance method
def update_wrapper(wrapper, wrapped, *a, **ka):
    try: functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError: pass



# These helpers are used at module level and need to be defined first.
# And yes, I know PEP-8, but sometimes a lower-case classname makes more sense.

def depr(message, hard=False):
    warnings.warn(message, DeprecationWarning, stacklevel=3)

def makelist(data): # This is just to handy
    if isinstance(data, (tuple, list, set, dict)): return list(data)
    elif data: return [data]
    else: return []


class DictProperty(object):
    ''' Property that maps to a key in a local dict-like attribute. '''
    def __init__(self, attr, key=None, read_only=False):
        self.attr, self.key, self.read_only = attr, key, read_only

    def __call__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter, self.key = func, self.key or func.__name__
        return self

    def __get__(self, obj, cls):
        if obj is None: return self
        key, storage = self.key, getattr(obj, self.attr)
        if key not in storage: storage[key] = self.getter(obj)
        return storage[key]

    def __set__(self, obj, value):
        if self.read_only: raise AttributeError("Read-Only property.")
        getattr(obj, self.attr)[self.key] = value

    def __delete__(self, obj):
        if self.read_only: raise AttributeError("Read-Only property.")
        del getattr(obj, self.attr)[self.key]


class cached_property(object):
    ''' A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. '''

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class lazy_attribute(object):
    ''' A property that caches itself to the class object. '''
    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value






###############################################################################
# Exceptions and Events ########################################################
###############################################################################


class BottleException(Exception):
    """ A base class for exceptions used by bottle. """
    pass






###############################################################################
# Routing ######################################################################
###############################################################################


class RouteError(BottleException):
    """ This is a base class for all routing related exceptions """


class RouteReset(BottleException):
    """ If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. """

class RouterUnknownModeError(RouteError): pass


class RouteSyntaxError(RouteError):
    """ The route parser found something not supported by this router. """


class RouteBuildError(RouteError):
    """ The route could not be built. """


def _re_flatten(p):
    ''' Turn all capturing groups in a regular expression pattern into
        non-capturing groups. '''
    if '(' not in p: return p
    return re.sub(r'(\\*)(\(\?P<[^>]+>|\((?!\?))',
        lambda m: m.group(0) if len(m.group(1)) % 2 else m.group(1) + '(?:', p)


class Router(object):
    ''' A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    '''

    default_pattern = '[^/]+'
    default_filter  = 're'

    #: The current CPython regexp implementation does not allow more
    #: than 99 matching groups per regular expression.
    _MAX_GROUPS_PER_PATTERN = 99

    def __init__(self, strict=False):
        self.rules    = [] # All rules in order
        self._groups  = {} # index of regexes to find them in dyna_routes
        self.builder  = {} # Data structure for the url builder
        self.static   = {} # Search structure for static routes
        self.dyna_routes   = {}
        self.dyna_regexes  = {} # Search structure for dynamic routes
        #: If true, static routes are no longer checked first.
        self.strict_order = strict
        self.filters = {
            're':    lambda conf:
                (_re_flatten(conf or self.default_pattern), None, None),
            'int':   lambda conf: (r'-?\d+', int, lambda x: str(int(x))),
            'float': lambda conf: (r'-?[\d.]+', float, lambda x: str(float(x))),
            'path':  lambda conf: (r'.+?', None, None)}

    def add_filter(self, name, func):
        ''' Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. '''
        self.filters[name] = func

    rule_syntax = re.compile('(\\\\*)'\
        '(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)'\
          '|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)'\
            '(?::((?:\\\\.|[^\\\\>]+)+)?)?)?>))')

    def _itertokens(self, rule):
        offset, prefix = 0, ''
        for match in self.rule_syntax.finditer(rule):
            prefix += rule[offset:match.start()]
            g = match.groups()
            if len(g[0])%2: # Escaped wildcard
                prefix += match.group(0)[len(g[0]):]
                offset = match.end()
                continue
            if prefix:
                yield prefix, None, None
            name, filtr, conf = g[4:7] if g[2] is None else g[1:4]
            yield name, filtr or 'default', conf or None
            offset, prefix = match.end(), ''
        if offset <= len(rule) or prefix:
            yield prefix+rule[offset:], None, None

    def add(self, rule, method, target, name=None):
        ''' Add a new rule or replace the target for an existing rule. '''
        anons     = 0    # Number of anonymous wildcards found
        keys      = []   # Names of keys
        pattern   = ''   # Regular expression pattern with named groups
        filters   = []   # Lists of wildcard input filters
        builder   = []   # Data structure for the URL builder
        is_static = True

        for key, mode, conf in self._itertokens(rule):
            if mode:
                is_static = False
                if mode == 'default': mode = self.default_filter
                mask, in_filter, out_filter = self.filters[mode](conf)
                if not key:
                    pattern += '(?:%s)' % mask
                    key = 'anon%d' % anons
                    anons += 1
                else:
                    pattern += '(?P<%s>%s)' % (key, mask)
                    keys.append(key)
                if in_filter: filters.append((key, in_filter))
                builder.append((key, out_filter or str))
            elif key:
                pattern += re.escape(key)
                builder.append((None, key))

        self.builder[rule] = builder
        if name: self.builder[name] = builder

        if is_static and not self.strict_order:
            self.static.setdefault(method, {})
            self.static[method][self.build(rule)] = (target, None)
            return

        try:
            re_pattern = re.compile('^(%s)$' % pattern)
            re_match = re_pattern.match
        except re.error:
            raise RouteSyntaxError("Could not add Route: %s (%s)" % (rule, _e()))

        if filters:
            def getargs(path):
                url_args = re_match(path).groupdict()
                for name, wildcard_filter in filters:
                    try:
                        url_args[name] = wildcard_filter(url_args[name])
                    except ValueError:
                        raise HTTPError(400, 'Path has wrong format.')
                return url_args
        elif re_pattern.groupindex:
            def getargs(path):
                return re_match(path).groupdict()
        else:
            getargs = None

        flatpat = _re_flatten(pattern)
        whole_rule = (rule, flatpat, target, getargs)

        if (flatpat, method) in self._groups:
            if DEBUG:
                msg = 'Route <%s %s> overwrites a previously defined route'
                warnings.warn(msg % (method, rule), RuntimeWarning)
            self.dyna_routes[method][self._groups[flatpat, method]] = whole_rule
        else:
            self.dyna_routes.setdefault(method, []).append(whole_rule)
            self._groups[flatpat, method] = len(self.dyna_routes[method]) - 1

        self._compile(method)

    def _compile(self, method):
        all_rules = self.dyna_routes[method]
        comborules = self.dyna_regexes[method] = []
        maxgroups = self._MAX_GROUPS_PER_PATTERN
        for x in range(0, len(all_rules), maxgroups):
            some = all_rules[x:x+maxgroups]
            combined = (flatpat for (_, flatpat, _, _) in some)
            combined = '|'.join('(^%s$)' % flatpat for flatpat in combined)
            combined = re.compile(combined).match
            rules = [(target, getargs) for (_, _, target, getargs) in some]
            comborules.append((combined, rules))

    def build(self, _name, *anons, **query):
        ''' Build an URL by filling the wildcards in a rule. '''
        builder = self.builder.get(_name)
        if not builder: raise RouteBuildError("No route with that name.", _name)
        try:
            for i, value in enumerate(anons): query['anon%d'%i] = value
            url = ''.join([f(query.pop(n)) if n else f for (n,f) in builder])
            return url if not query else url+'?'+urlencode(query)
        except KeyError:
            raise RouteBuildError('Missing URL argument: %r' % _e().args[0])

    def match(self, environ):
        ''' Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). '''
        verb = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'] or '/'
        target = None
        if verb == 'HEAD':
            methods = ['PROXY', verb, 'GET', 'ANY']
        else:
            methods = ['PROXY', verb, 'ANY']

        for method in methods:
            if method in self.static and path in self.static[method]:
                target, getargs = self.static[method][path]
                return target, getargs(path) if getargs else {}
            elif method in self.dyna_regexes:
                for combined, rules in self.dyna_regexes[method]:
                    match = combined(path)
                    if match:
                        target, getargs = rules[match.lastindex - 1]
                        return target, getargs(path) if getargs else {}

        # No matching route found. Collect alternative methods for 405 response
        allowed = set([])
        nocheck = set(methods)
        for method in set(self.static) - nocheck:
            if path in self.static[method]:
                allowed.add(verb)
        for method in set(self.dyna_regexes) - allowed - nocheck:
            for combined, rules in self.dyna_regexes[method]:
                match = combined(path)
                if match:
                    allowed.add(method)
        if allowed:
            allow_header = ",".join(sorted(allowed))
            raise HTTPError(405, "Method not allowed.", Allow=allow_header)

        # No matching route and no alternative method found. We give up
        raise HTTPError(404, "Not found: " + repr(path))






class Route(object):
    ''' This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    '''

    def __init__(self, app, rule, method, callback, name=None,
                 plugins=None, skiplist=None, **config):
        #: The application this route is installed to.
        self.app = app
        #: The path-rule string (e.g. ``/wiki/:page``).
        self.rule = rule
        #: The HTTP method as a string (e.g. ``GET``).
        self.method = method
        #: The original callback with no plugins applied. Useful for introspection.
        self.callback = callback
        #: The name of the route (if specified) or ``None``.
        self.name = name or None
        #: A list of route-specific plugins (see :meth:`Bottle.route`).
        self.plugins = plugins or []
        #: A list of plugins to not apply to this route (see :meth:`Bottle.route`).
        self.skiplist = skiplist or []
        #: Additional keyword arguments passed to the :meth:`Bottle.route`
        #: decorator are stored in this dictionary. Used for route-specific
        #: plugin configuration and meta-data.
        self.config = ConfigDict().load_dict(config, make_namespaces=True)

    def __call__(self, *a, **ka):
        depr("Some APIs changed to return Route() instances instead of"\
             " callables. Make sure to use the Route.call method and not to"\
             " call Route instances directly.") #0.12
        return self.call(*a, **ka)

    @cached_property
    def call(self):
        ''' The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.'''
        return self._make_callback()

    def reset(self):
        ''' Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. '''
        self.__dict__.pop('call', None)

    def prepare(self):
        ''' Do all on-demand work immediately (useful for debugging).'''
        self.call

    @property
    def _context(self):
        depr('Switch to Plugin API v2 and access the Route object directly.')  #0.12
        return dict(rule=self.rule, method=self.method, callback=self.callback,
                    name=self.name, app=self.app, config=self.config,
                    apply=self.plugins, skip=self.skiplist)

    def all_plugins(self):
        ''' Yield all Plugins affecting this route. '''
        unique = set()
        for p in reversed(self.app.plugins + self.plugins):
            if True in self.skiplist: break
            name = getattr(p, 'name', False)
            if name and (name in self.skiplist or name in unique): continue
            if p in self.skiplist or type(p) in self.skiplist: continue
            if name: unique.add(name)
            yield p

    def _make_callback(self):
        callback = self.callback
        for plugin in self.all_plugins():
            try:
                if hasattr(plugin, 'apply'):
                    api = getattr(plugin, 'api', 1)
                    context = self if api > 1 else self._context
                    callback = plugin.apply(callback, context)
                else:
                    callback = plugin(callback)
            except RouteReset: # Try again with changed configuration.
                return self._make_callback()
            if not callback is self.callback:
                update_wrapper(callback, self.callback)
        return callback

    def get_undecorated_callback(self):
        ''' Return the callback. If the callback is a decorated function, try to
            recover the original function. '''
        func = self.callback
        func = getattr(func, '__func__' if py3k else 'im_func', func)
        closure_attr = '__closure__' if py3k else 'func_closure'
        while hasattr(func, closure_attr) and getattr(func, closure_attr):
            func = getattr(func, closure_attr)[0].cell_contents
        return func

    def get_callback_args(self):
        ''' Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. '''
        return getargspec(self.get_undecorated_callback())[0]

    def get_config(self, key, default=None):
        ''' Lookup a config field and return its value, first checking the
            route.config, then route.app.config.'''
        for conf in (self.config, self.app.conifg):
            if key in conf: return conf[key]
        return default

    def __repr__(self):
        cb = self.get_undecorated_callback()
        return '<%s %r %r>' % (self.method, self.rule, cb)






###############################################################################
# Application Object ###########################################################
###############################################################################


class Bottle(object):
    """ Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    """

    def __init__(self, catchall=True, autojson=True):

        #: A :class:`ConfigDict` for app specific configuration.
        self.config = ConfigDict()
        self.config._on_change = functools.partial(self.trigger_hook, 'config')
        self.config.meta_set('autojson', 'validate', bool)
        self.config.meta_set('catchall', 'validate', bool)
        self.config['catchall'] = catchall
        self.config['autojson'] = autojson

        #: A :class:`ResourceManager` for application files
        self.resources = ResourceManager()

        self.routes = [] # List of installed :class:`Route` instances.
        self.router = Router() # Maps requests to :class:`Route` instances.
        self.error_handler = {}

        # Core plugins
        self.plugins = [] # List of installed plugins.
        if self.config['autojson']:
            self.install(JSONPlugin())
        self.install(TemplatePlugin())

    #: If true, most exceptions are caught and returned as :exc:`HTTPError`
    catchall = DictProperty('config', 'catchall')

    __hook_names = 'before_request', 'after_request', 'app_reset', 'config'
    __hook_reversed = 'after_request'

    @cached_property
    def _hooks(self):
        return dict((name, []) for name in self.__hook_names)

    def add_hook(self, name, func):
        ''' Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        '''
        if name in self.__hook_reversed:
            self._hooks[name].insert(0, func)
        else:
            self._hooks[name].append(func)

    def remove_hook(self, name, func):
        ''' Remove a callback from a hook. '''
        if name in self._hooks and func in self._hooks[name]:
            self._hooks[name].remove(func)
            return True

    def trigger_hook(self, __name, *args, **kwargs):
        ''' Trigger a hook and return a list of results. '''
        return [hook(*args, **kwargs) for hook in self._hooks[__name][:]]

    def hook(self, name):
        """ Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details."""
        def decorator(func):
            self.add_hook(name, func)
            return func
        return decorator

    def mount(self, prefix, app, **options):
        ''' Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        '''
        if isinstance(app, basestring):
            depr('Parameter order of Bottle.mount() changed.', True) # 0.10

        segments = [p for p in prefix.split('/') if p]
        if not segments: raise ValueError('Empty path prefix.')
        path_depth = len(segments)

        def mountpoint_wrapper():
            try:
                request.path_shift(path_depth)
                rs = HTTPResponse([])
                def start_response(status, headerlist, exc_info=None):
                    if exc_info:
                        try:
                            _raise(*exc_info)
                        finally:
                            exc_info = None
                    rs.status = status
                    for name, value in headerlist: rs.add_header(name, value)
                    return rs.body.append
                body = app(request.environ, start_response)
                if body and rs.body: body = itertools.chain(rs.body, body)
                rs.body = body or rs.body
                return rs
            finally:
                request.path_shift(-path_depth)

        options.setdefault('skip', True)
        options.setdefault('method', 'PROXY')
        options.setdefault('mountpoint', {'prefix': prefix, 'target': app})
        options['callback'] = mountpoint_wrapper

        self.route('/%s/<:re:.*>' % '/'.join(segments), **options)
        if not prefix.endswith('/'):
            self.route('/' + '/'.join(segments), **options)

    def merge(self, routes):
        ''' Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. '''
        if isinstance(routes, Bottle):
            routes = routes.routes
        for route in routes:
            self.add_route(route)

    def install(self, plugin):
        ''' Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        '''
        if hasattr(plugin, 'setup'): plugin.setup(self)
        if not callable(plugin) and not hasattr(plugin, 'apply'):
            raise TypeError("Plugins must be callable or implement .apply()")
        self.plugins.append(plugin)
        self.reset()
        return plugin

    def uninstall(self, plugin):
        ''' Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. '''
        removed, remove = [], plugin
        for i, plugin in list(enumerate(self.plugins))[::-1]:
            if remove is True or remove is plugin or remove is type(plugin) \
            or getattr(plugin, 'name', True) == remove:
                removed.append(plugin)
                del self.plugins[i]
                if hasattr(plugin, 'close'): plugin.close()
        if removed: self.reset()
        return removed

    def reset(self, route=None):
        ''' Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. '''
        if route is None: routes = self.routes
        elif isinstance(route, Route): routes = [route]
        else: routes = [self.routes[route]]
        for route in routes: route.reset()
        if DEBUG:
            for route in routes: route.prepare()
        self.trigger_hook('app_reset')

    def close(self):
        ''' Close the application and all installed plugins. '''
        for plugin in self.plugins:
            if hasattr(plugin, 'close'): plugin.close()
        self.stopped = True

    def run(self, **kwargs):
        ''' Calls :func:`run` with the same parameters. '''
        run(self, **kwargs)

    def match(self, environ):
        """ Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match."""
        return self.router.match(environ)

    def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        scriptname = request.environ.get('SCRIPT_NAME', '').strip('/') + '/'
        location = self.router.build(routename, **kargs).lstrip('/')
        return urljoin(urljoin('/', scriptname), location)

    def add_route(self, route):
        ''' Add a route object, but do not change the :data:`Route.app`
            attribute.'''
        self.routes.append(route)
        self.router.add(route.rule, route.method, route, name=route.name)
        if DEBUG: route.prepare()

    def route(self, path=None, method='GET', callback=None, name=None,
              apply=None, skip=None, **config):
        """ A decorator to bind a function to a request URL. Example::

                @app.route('/hello/:name')
                def hello(name):
                    return 'Hello %s' % name

            The ``:name`` part is a wildcard. See :class:`Router` for syntax
            details.

            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.

            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        """
        if callable(path): path, callback = None, path
        plugins = makelist(apply)
        skiplist = makelist(skip)
        def decorator(callback):
            # TODO: Documentation and tests
            if isinstance(callback, basestring): callback = load(callback)
            for rule in makelist(path) or yieldroutes(callback):
                for verb in makelist(method):
                    verb = verb.upper()
                    route = Route(self, rule, verb, callback, name=name,
                                  plugins=plugins, skiplist=skiplist, **config)
                    self.add_route(route)
            return callback
        return decorator(callback) if callback else decorator

    def get(self, path=None, method='GET', **options):
        """ Equals :meth:`route`. """
        return self.route(path, method, **options)

    def post(self, path=None, method='POST', **options):
        """ Equals :meth:`route` with a ``POST`` method parameter. """
        return self.route(path, method, **options)

    def put(self, path=None, method='PUT', **options):
        """ Equals :meth:`route` with a ``PUT`` method parameter. """
        return self.route(path, method, **options)

    def delete(self, path=None, method='DELETE', **options):
        """ Equals :meth:`route` with a ``DELETE`` method parameter. """
        return self.route(path, method, **options)

    def error(self, code=500):
        """ Decorator: Register an output handler for a HTTP error code"""
        def wrapper(handler):
            self.error_handler[int(code)] = handler
            return handler
        return wrapper

    def default_error_handler(self, res):
        return tob(template(ERROR_PAGE_TEMPLATE, e=res))

    def _handle(self, environ):
        path = environ['bottle.raw_path'] = environ['PATH_INFO']
        if py3k:
            try:
                environ['PATH_INFO'] = path.encode('latin1').decode('utf8')
            except UnicodeError:
                return HTTPError(400, 'Invalid path string. Expected UTF-8')

        try:
            environ['bottle.app'] = self
            request.bind(environ)
            response.bind()
            try:
                self.trigger_hook('before_request')
                route, args = self.router.match(environ)
                environ['route.handle'] = route
                environ['bottle.route'] = route
                environ['route.url_args'] = args
                return route.call(**args)
            finally:
                self.trigger_hook('after_request')

        except HTTPResponse:
            return _e()
        except RouteReset:
            route.reset()
            return self._handle(environ)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            stacktrace = format_exc()
            environ['wsgi.errors'].write(stacktrace)
            return HTTPError(500, "Internal Server Error", _e(), stacktrace)

    def _cast(self, out, peek=None):
        """ Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        """

        # Empty output is done here
        if not out:
            if 'Content-Length' not in response:
                response['Content-Length'] = 0
            return []
        # Join lists of byte or unicode strings. Mixed lists are NOT supported
        if isinstance(out, (tuple, list))\
        and isinstance(out[0], (bytes, unicode)):
            out = out[0][0:0].join(out) # b'abc'[0:0] -> b''
        # Encode unicode strings
        if isinstance(out, unicode):
            out = out.encode(response.charset)
        # Byte Strings are just returned
        if isinstance(out, bytes):
            if 'Content-Length' not in response:
                response['Content-Length'] = len(out)
            return [out]
        # HTTPError or HTTPException (recursive, because they may wrap anything)
        # TODO: Handle these explicitly in handle() or make them iterable.
        if isinstance(out, HTTPError):
            out.apply(response)
            out = self.error_handler.get(out.status_code, self.default_error_handler)(out)
            return self._cast(out)
        if isinstance(out, HTTPResponse):
            out.apply(response)
            return self._cast(out.body)

        # File-like objects.
        if hasattr(out, 'read'):
            if 'wsgi.file_wrapper' in request.environ:
                return request.environ['wsgi.file_wrapper'](out)
            elif hasattr(out, 'close') or not hasattr(out, '__iter__'):
                return WSGIFileWrapper(out)

        # Handle Iterables. We peek into them to detect their inner type.
        try:
            iout = iter(out)
            first = next(iout)
            while not first:
                first = next(iout)
        except StopIteration:
            return self._cast('')
        except HTTPResponse:
            first = _e()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            first = HTTPError(500, 'Unhandled exception', _e(), format_exc())

        # These are the inner types allowed in iterator or generator objects.
        if isinstance(first, HTTPResponse):
            return self._cast(first)
        elif isinstance(first, bytes):
            new_iter = itertools.chain([first], iout)
        elif isinstance(first, unicode):
            encoder = lambda x: x.encode(response.charset)
            new_iter = imap(encoder, itertools.chain([first], iout))
        else:
            msg = 'Unsupported response type: %s' % type(first)
            return self._cast(HTTPError(500, msg))
        if hasattr(out, 'close'):
            new_iter = _closeiter(new_iter, out.close)
        return new_iter

    def wsgi(self, environ, start_response):
        """ The bottle WSGI-interface. """
        try:
            out = self._cast(self._handle(environ))
            # rfc2616 section 4.3
            if response._status_code in (100, 101, 204, 304)\
            or environ['REQUEST_METHOD'] == 'HEAD':
                if hasattr(out, 'close'): out.close()
                out = []
            start_response(response._status_line, response.headerlist)
            return out
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            err = '<h1>Critical error while processing request: %s</h1>' \
                  % html_escape(environ.get('PATH_INFO', '/'))
            if DEBUG:
                err += '<h2>Error:</h2>\n<pre>\n%s\n</pre>\n' \
                       '<h2>Traceback:</h2>\n<pre>\n%s\n</pre>\n' \
                       % (html_escape(repr(_e())), html_escape(format_exc()))
            environ['wsgi.errors'].write(err)
            headers = [('Content-Type', 'text/html; charset=UTF-8')]
            start_response('500 INTERNAL SERVER ERROR', headers, sys.exc_info())
            return [tob(err)]

    def __call__(self, environ, start_response):
        ''' Each instance of :class:'Bottle' is a WSGI application. '''
        return self.wsgi(environ, start_response)






###############################################################################
# HTTP and WSGI Tools ##########################################################
###############################################################################

class BaseRequest(object):
    """ A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    """

    __slots__ = ('environ')

    #: Maximum size of memory buffer for :attr:`body` in bytes.
    MEMFILE_MAX = 102400

    def __init__(self, environ=None):
        """ Wrap a WSGI environ dictionary. """
        #: The wrapped WSGI environ dictionary. This is the only real attribute.
        #: All other attributes actually are read-only properties.
        self.environ = {} if environ is None else environ
        self.environ['bottle.request'] = self

    @DictProperty('environ', 'bottle.app', read_only=True)
    def app(self):
        ''' Bottle application handling this request. '''
        raise RuntimeError('This request is not connected to an application.')

    @DictProperty('environ', 'bottle.route', read_only=True)
    def route(self):
        """ The bottle :class:`Route` object that matches this request. """
        raise RuntimeError('This request is not connected to a route.')

    @DictProperty('environ', 'route.url_args', read_only=True)
    def url_args(self):
        """ The arguments extracted from the URL. """
        raise RuntimeError('This request is not connected to a route.')

    @property
    def path(self):
        ''' The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). '''
        return '/' + self.environ.get('PATH_INFO','').lstrip('/')

    @property
    def method(self):
        ''' The ``REQUEST_METHOD`` value as an uppercase string. '''
        return self.environ.get('REQUEST_METHOD', 'GET').upper()

    @DictProperty('environ', 'bottle.request.headers', read_only=True)
    def headers(self):
        ''' A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. '''
        return WSGIHeaderDict(self.environ)

    def get_header(self, name, default=None):
        ''' Return the value of a request header, or a given default value. '''
        return self.headers.get(name, default)

    @DictProperty('environ', 'bottle.request.cookies', read_only=True)
    def cookies(self):
        """ Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. """
        cookies = SimpleCookie(self.environ.get('HTTP_COOKIE','')).values()
        return FormsDict((c.key, c.value) for c in cookies)

    def get_cookie(self, key, default=None, secret=None):
        """ Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. """
        value = self.cookies.get(key)
        if secret and value:
            dec = cookie_decode(value, secret) # (key, value) tuple or None
            return dec[1] if dec and dec[0] == key else default
        return value or default

    @DictProperty('environ', 'bottle.request.query', read_only=True)
    def query(self):
        ''' The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. '''
        get = self.environ['bottle.get'] = FormsDict()
        pairs = _parse_qsl(self.environ.get('QUERY_STRING', ''))
        for key, value in pairs:
            get[key] = value
        return get

    @DictProperty('environ', 'bottle.request.forms', read_only=True)
    def forms(self):
        """ Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. """
        forms = FormsDict()
        for name, item in self.POST.allitems():
            if not isinstance(item, FileUpload):
                forms[name] = item
        return forms

    @DictProperty('environ', 'bottle.request.params', read_only=True)
    def params(self):
        """ A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. """
        params = FormsDict()
        for key, value in self.query.allitems():
            params[key] = value
        for key, value in self.forms.allitems():
            params[key] = value
        return params

    @DictProperty('environ', 'bottle.request.files', read_only=True)
    def files(self):
        """ File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        """
        files = FormsDict()
        for name, item in self.POST.allitems():
            if isinstance(item, FileUpload):
                files[name] = item
        return files

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        ''' If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. '''
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype == 'application/json':
            b = self._get_body_string()
            if not b:
                return None
            return json_loads(b)
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    def _iter_chunked(self, read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        read_func = self.environ['wsgi.input'].read
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self):
        ''' read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. '''
        clen = self.content_length
        if clen > self.MEMFILE_MAX:
            raise HTTPError(413, 'Request to large')
        if clen < 0: clen = self.MEMFILE_MAX + 1
        data = self.body.read(clen)
        if len(data) > self.MEMFILE_MAX: # Fail fast
            raise HTTPError(413, 'Request to large')
        return data

    @property
    def body(self):
        """ The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. """
        self._body.seek(0)
        return self._body

    @property
    def chunked(self):
        ''' True if Chunked transfer encoding was. '''
        return 'chunked' in self.environ.get('HTTP_TRANSFER_ENCODING', '').lower()

    #: An alias for :attr:`query`.
    GET = query

    @DictProperty('environ', 'bottle.request.post', read_only=True)
    def POST(self):
        """ The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        """
        post = FormsDict()
        # We default to application/x-www-form-urlencoded for everything that
        # is not multipart and take the fast path (also: 3.1 workaround)
        if not self.content_type.startswith('multipart/'):
            pairs = _parse_qsl(tonat(self._get_body_string(), 'latin1'))
            for key, value in pairs:
                post[key] = value
            return post

        safe_env = {'QUERY_STRING':''} # Build a safe environment for cgi
        for key in ('REQUEST_METHOD', 'CONTENT_TYPE', 'CONTENT_LENGTH'):
            if key in self.environ: safe_env[key] = self.environ[key]
        args = dict(fp=self.body, environ=safe_env, keep_blank_values=True)
        if py31:
            args['fp'] = NCTextIOWrapper(args['fp'], encoding='utf8',
                                         newline='\n')
        elif py3k:
            args['encoding'] = 'utf8'
        data = cgi.FieldStorage(**args)
        self['_cgi.FieldStorage'] = data #http://bugs.python.org/issue18394#msg207958
        data = data.list or []
        for item in data:
            if item.filename:
                post[item.name] = FileUpload(item.file, item.name,
                                             item.filename, item.headers)
            else:
                post[item.name] = item.value
        return post

    @property
    def url(self):
        """ The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. """
        return self.urlparts.geturl()

    @DictProperty('environ', 'bottle.request.urlparts', read_only=True)
    def urlparts(self):
        ''' The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. '''
        env = self.environ
        http = env.get('HTTP_X_FORWARDED_PROTO') or env.get('wsgi.url_scheme', 'http')
        host = env.get('HTTP_X_FORWARDED_HOST') or env.get('HTTP_HOST')
        if not host:
            # HTTP 1.1 requires a Host-header. This is for HTTP/1.0 clients.
            host = env.get('SERVER_NAME', '127.0.0.1')
            port = env.get('SERVER_PORT')
            if port and port != ('80' if http == 'http' else '443'):
                host += ':' + port
        path = urlquote(self.fullpath)
        return UrlSplitResult(http, host, path, env.get('QUERY_STRING'), '')

    @property
    def fullpath(self):
        """ Request path including :attr:`script_name` (if present). """
        return urljoin(self.script_name, self.path.lstrip('/'))

    @property
    def query_string(self):
        """ The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. """
        return self.environ.get('QUERY_STRING', '')

    @property
    def script_name(self):
        ''' The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. '''
        script_name = self.environ.get('SCRIPT_NAME', '').strip('/')
        return '/' + script_name + '/' if script_name else '/'

    def path_shift(self, shift=1):
        ''' Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        '''
        script = self.environ.get('SCRIPT_NAME','/')
        self['SCRIPT_NAME'], self['PATH_INFO'] = path_shift(script, self.path, shift)

    @property
    def content_length(self):
        ''' The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. '''
        return int(self.environ.get('CONTENT_LENGTH') or -1)

    @property
    def content_type(self):
        ''' The Content-Type header as a lowercase-string (default: empty). '''
        return self.environ.get('CONTENT_TYPE', '').lower()

    @property
    def is_xhr(self):
        ''' True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). '''
        requested_with = self.environ.get('HTTP_X_REQUESTED_WITH','')
        return requested_with.lower() == 'xmlhttprequest'

    @property
    def is_ajax(self):
        ''' Alias for :attr:`is_xhr`. "Ajax" is not the right term. '''
        return self.is_xhr

    @property
    def auth(self):
        """ HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. """
        basic = parse_auth(self.environ.get('HTTP_AUTHORIZATION',''))
        if basic: return basic
        ruser = self.environ.get('REMOTE_USER')
        if ruser: return (ruser, None)
        return None

    @property
    def remote_route(self):
        """ A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. """
        proxy = self.environ.get('HTTP_X_FORWARDED_FOR')
        if proxy: return [ip.strip() for ip in proxy.split(',')]
        remote = self.environ.get('REMOTE_ADDR')
        return [remote] if remote else []

    @property
    def remote_addr(self):
        """ The client IP as a string. Note that this information can be forged
            by malicious clients. """
        route = self.remote_route
        return route[0] if route else None

    def copy(self):
        """ Return a new :class:`Request` with a shallow :attr:`environ` copy. """
        return Request(self.environ.copy())

    def get(self, value, default=None): return self.environ.get(value, default)
    def __getitem__(self, key): return self.environ[key]
    def __delitem__(self, key): self[key] = ""; del(self.environ[key])
    def __iter__(self): return iter(self.environ)
    def __len__(self): return len(self.environ)
    def keys(self): return self.environ.keys()
    def __setitem__(self, key, value):
        """ Change an environ value and clear all caches that depend on it. """

        if self.environ.get('bottle.request.readonly'):
            raise KeyError('The environ dictionary is read-only.')

        self.environ[key] = value
        todelete = ()

        if key == 'wsgi.input':
            todelete = ('body', 'forms', 'files', 'params', 'post', 'json')
        elif key == 'QUERY_STRING':
            todelete = ('query', 'params')
        elif key.startswith('HTTP_'):
            todelete = ('headers', 'cookies')

        for key in todelete:
            self.environ.pop('bottle.request.'+key, None)

    def __repr__(self):
        return '<%s: %s %s>' % (self.__class__.__name__, self.method, self.url)

    def __getattr__(self, name):
        ''' Search in self.environ for additional user defined attributes. '''
        try:
            var = self.environ['bottle.request.ext.%s'%name]
            return var.__get__(self) if hasattr(var, '__get__') else var
        except KeyError:
            raise AttributeError('Attribute %r not defined.' % name)

    def __setattr__(self, name, value):
        if name == 'environ': return object.__setattr__(self, name, value)
        self.environ['bottle.request.ext.%s'%name] = value




def _hkey(s):
    return s.title().replace('_','-')


class HeaderProperty(object):
    def __init__(self, name, reader=None, writer=str, default=''):
        self.name, self.default = name, default
        self.reader, self.writer = reader, writer
        self.__doc__ = 'Current value of the %r header.' % name.title()

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.headers.get(self.name, self.default)
        return self.reader(value) if self.reader else value

    def __set__(self, obj, value):
        obj.headers[self.name] = self.writer(value)

    def __delete__(self, obj):
        del obj.headers[self.name]


class BaseResponse(object):
    """ Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    """

    default_status = 200
    default_content_type = 'text/html; charset=UTF-8'

    # Header blacklist for specific response codes
    # (rfc2616 section 10.2.3 and 10.3.5)
    bad_headers = {
        204: set(('Content-Type',)),
        304: set(('Allow', 'Content-Encoding', 'Content-Language',
                  'Content-Length', 'Content-Range', 'Content-Type',
                  'Content-Md5', 'Last-Modified'))}

    def __init__(self, body='', status=None, headers=None, **more_headers):
        self._cookies = None
        self._headers = {}
        self.body = body
        self.status = status or self.default_status
        if headers:
            if isinstance(headers, dict):
                headers = headers.items()
            for name, value in headers:
                self.add_header(name, value)
        if more_headers:
            for name, value in more_headers.items():
                self.add_header(name, value)

    def copy(self, cls=None):
        ''' Returns a copy of self. '''
        cls = cls or BaseResponse
        assert issubclass(cls, BaseResponse)
        copy = cls()
        copy.status = self.status
        copy._headers = dict((k, v[:]) for (k, v) in self._headers.items())
        if self._cookies:
            copy._cookies = SimpleCookie()
            copy._cookies.load(self._cookies.output(header=''))
        return copy

    def __iter__(self):
        return iter(self.body)

    def close(self):
        if hasattr(self.body, 'close'):
            self.body.close()

    @property
    def status_line(self):
        ''' The HTTP status line as a string (e.g. ``404 Not Found``).'''
        return self._status_line

    @property
    def status_code(self):
        ''' The HTTP status code as an integer (e.g. 404).'''
        return self._status_code

    def _set_status(self, status):
        if isinstance(status, int):
            code, status = status, _HTTP_STATUS_LINES.get(status)
        elif ' ' in status:
            status = status.strip()
            code   = int(status.split()[0])
        else:
            raise ValueError('String status line without a reason phrase.')
        if not 100 <= code <= 999: raise ValueError('Status code out of range.')
        self._status_code = code
        self._status_line = str(status or ('%d Unknown' % code))

    def _get_status(self):
        return self._status_line

    status = property(_get_status, _set_status, None,
        ''' A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. ''')
    del _get_status, _set_status

    @property
    def headers(self):
        ''' An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. '''
        hdict = HeaderDict()
        hdict.dict = self._headers
        return hdict

    def __contains__(self, name): return _hkey(name) in self._headers
    def __delitem__(self, name):  del self._headers[_hkey(name)]
    def __getitem__(self, name):  return self._headers[_hkey(name)][-1]
    def __setitem__(self, name, value): self._headers[_hkey(name)] = [str(value)]

    def get_header(self, name, default=None):
        ''' Return the value of a previously defined header. If there is no
            header with that name, return a default value. '''
        return self._headers.get(_hkey(name), [default])[-1]

    def set_header(self, name, value):
        ''' Create a new response header, replacing any previously defined
            headers with the same name. '''
        self._headers[_hkey(name)] = [str(value)]

    def add_header(self, name, value):
        ''' Add an additional response header, not removing duplicates. '''
        self._headers.setdefault(_hkey(name), []).append(str(value))

    def iter_headers(self):
        ''' Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. '''
        return self.headerlist

    @property
    def headerlist(self):
        ''' WSGI conform list of (header, value) tuples. '''
        out = []
        headers = list(self._headers.items())
        if 'Content-Type' not in self._headers:
            headers.append(('Content-Type', [self.default_content_type]))
        if self._status_code in self.bad_headers:
            bad_headers = self.bad_headers[self._status_code]
            headers = [h for h in headers if h[0] not in bad_headers]
        out += [(name, val) for name, vals in headers for val in vals]
        if self._cookies:
            for c in self._cookies.values():
                out.append(('Set-Cookie', c.OutputString()))
        return out

    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int)
    expires = HeaderProperty('Expires',
        reader=lambda x: datetime.utcfromtimestamp(parse_date(x)),
        writer=lambda x: http_date(x))

    @property
    def charset(self, default='UTF-8'):
        """ Return the charset specified in the content-type header (default: utf8). """
        if 'charset=' in self.content_type:
            return self.content_type.split('charset=')[-1].split(';')[0].strip()
        return default

    def set_cookie(self, name, value, secret=None, **options):
        ''' Create a new cookie or replace an old one. If the `secret` parameter is
            set, create a `Signed Cookie` (described below).

            :param name: the name of the cookie.
            :param value: the value of the cookie.
            :param secret: a signature key required for signed cookies.

            Additionally, this method accepts all RFC 2109 attributes that are
            supported by :class:`cookie.Morsel`, including:

            :param max_age: maximum age in seconds. (default: None)
            :param expires: a datetime object or UNIX timestamp. (default: None)
            :param domain: the domain that is allowed to read the cookie.
              (default: current domain)
            :param path: limits the cookie to a given path (default: current path)
            :param secure: limit the cookie to HTTPS connections (default: off).
            :param httponly: prevents client-side javascript to read this cookie
              (default: off, requires Python 2.6 or newer).

            If neither `expires` nor `max_age` is set (default), the cookie will
            expire at the end of the browser session (as soon as the browser
            window is closed).

            Signed cookies may store any pickle-able object and are
            cryptographically signed to prevent manipulation. Keep in mind that
            cookies are limited to 4kb in most browsers.

            Warning: Signed cookies are not encrypted (the client can still see
            the content) and not copy-protected (the client can restore an old
            cookie). The main intention is to make pickling and unpickling
            save, not to store secret information at client side.
        '''
        if not self._cookies:
            self._cookies = SimpleCookie()

        if secret:
            value = touni(cookie_encode((name, value), secret))
        elif not isinstance(value, basestring):
            raise TypeError('Secret key missing for non-string Cookie.')

        if len(value) > 4096: raise ValueError('Cookie value to long.')
        self._cookies[name] = value

        for key, value in options.items():
            if key == 'max_age':
                if isinstance(value, timedelta):
                    value = value.seconds + value.days * 24 * 3600
            if key == 'expires':
                if isinstance(value, (datedate, datetime)):
                    value = value.timetuple()
                elif isinstance(value, (int, float)):
                    value = time.gmtime(value)
                value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
            self._cookies[name][key.replace('_', '-')] = value

    def delete_cookie(self, key, **kwargs):
        ''' Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. '''
        kwargs['max_age'] = -1
        kwargs['expires'] = 0
        self.set_cookie(key, '', **kwargs)

    def __repr__(self):
        out = ''
        for name, value in self.headerlist:
            out += '%s: %s\n' % (name.title(), value.strip())
        return out


def local_property(name=None):
    if name: depr('local_property() is deprecated and will be removed.') #0.12
    ls = threading.local()
    def fget(self):
        try: return ls.var
        except AttributeError:
            raise RuntimeError("Request context not initialized.")
    def fset(self, value): ls.var = value
    def fdel(self): del ls.var
    return property(fget, fset, fdel, 'Thread-local property')


class LocalRequest(BaseRequest):
    ''' A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). '''
    bind = BaseRequest.__init__
    environ = local_property()


class LocalResponse(BaseResponse):
    ''' A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    '''
    bind = BaseResponse.__init__
    _status_line = local_property()
    _status_code = local_property()
    _cookies     = local_property()
    _headers     = local_property()
    body         = local_property()


Request = BaseRequest
Response = BaseResponse


class HTTPResponse(Response, BottleException):
    def __init__(self, body='', status=None, headers=None, **more_headers):
        super(HTTPResponse, self).__init__(body, status, headers, **more_headers)

    def apply(self, response):
        response._status_code = self._status_code
        response._status_line = self._status_line
        response._headers = self._headers
        response._cookies = self._cookies
        response.body = self.body


class HTTPError(HTTPResponse):
    default_status = 500
    def __init__(self, status=None, body=None, exception=None, traceback=None,
                 **options):
        self.exception = exception
        self.traceback = traceback
        super(HTTPError, self).__init__(body, status, **options)





###############################################################################
# Plugins ######################################################################
###############################################################################

class PluginError(BottleException): pass


class JSONPlugin(object):
    name = 'json'
    api  = 2

    def __init__(self, json_dumps=json_dumps):
        self.json_dumps = json_dumps

    def apply(self, callback, route):
        dumps = self.json_dumps
        if not dumps: return callback
        def wrapper(*a, **ka):
            try:
                rv = callback(*a, **ka)
            except HTTPError:
                rv = _e()

            if isinstance(rv, dict):
                #Attempt to serialize, raises exception on failure
                json_response = dumps(rv)
                #Set content type only if serialization succesful
                response.content_type = 'application/json'
                return json_response
            elif isinstance(rv, HTTPResponse) and isinstance(rv.body, dict):
                rv.body = dumps(rv.body)
                rv.content_type = 'application/json'
            return rv

        return wrapper


class TemplatePlugin(object):
    ''' This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. '''
    name = 'template'
    api  = 2

    def apply(self, callback, route):
        conf = route.config.get('template')
        if isinstance(conf, (tuple, list)) and len(conf) == 2:
            return view(conf[0], **conf[1])(callback)
        elif isinstance(conf, str):
            return view(conf)(callback)
        else:
            return callback


#: Not a plugin, but part of the plugin API. TODO: Find a better place.
class _ImportRedirect(object):
    def __init__(self, name, impmask):
        ''' Create a virtual package that redirects imports (see PEP 302). '''
        self.name = name
        self.impmask = impmask
        self.module = sys.modules.setdefault(name, imp.new_module(name))
        self.module.__dict__.update({'__file__': __file__, '__path__': [],
                                    '__all__': [], '__loader__': self})
        sys.meta_path.append(self)

    def find_module(self, fullname, path=None):
        if '.' not in fullname: return
        packname = fullname.rsplit('.', 1)[0]
        if packname != self.name: return
        return self

    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        modname = fullname.rsplit('.', 1)[1]
        realname = self.impmask % modname
        __import__(realname)
        module = sys.modules[fullname] = sys.modules[realname]
        setattr(self.module, modname, module)
        module.__loader__ = self
        return module






###############################################################################
# Common Utilities #############################################################
###############################################################################


class MultiDict(DictMixin):
    """ This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    """

    def __init__(self, *a, **k):
        self.dict = dict((k, [v]) for (k, v) in dict(*a, **k).items())

    def __len__(self): return len(self.dict)
    def __iter__(self): return iter(self.dict)
    def __contains__(self, key): return key in self.dict
    def __delitem__(self, key): del self.dict[key]
    def __getitem__(self, key): return self.dict[key][-1]
    def __setitem__(self, key, value): self.append(key, value)
    def keys(self): return self.dict.keys()

    if py3k:
        def values(self): return (v[-1] for v in self.dict.values())
        def items(self): return ((k, v[-1]) for k, v in self.dict.items())
        def allitems(self):
            return ((k, v) for k, vl in self.dict.items() for v in vl)
        iterkeys = keys
        itervalues = values
        iteritems = items
        iterallitems = allitems

    else:
        def values(self): return [v[-1] for v in self.dict.values()]
        def items(self): return [(k, v[-1]) for k, v in self.dict.items()]
        def iterkeys(self): return self.dict.iterkeys()
        def itervalues(self): return (v[-1] for v in self.dict.itervalues())
        def iteritems(self):
            return ((k, v[-1]) for k, v in self.dict.iteritems())
        def iterallitems(self):
            return ((k, v) for k, vl in self.dict.iteritems() for v in vl)
        def allitems(self):
            return [(k, v) for k, vl in self.dict.iteritems() for v in vl]

    def get(self, key, default=None, index=-1, type=None):
        ''' Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        '''
        try:
            val = self.dict[key][index]
            return type(val) if type else val
        except Exception:
            pass
        return default

    def append(self, key, value):
        ''' Add a new value to the list of values for this key. '''
        self.dict.setdefault(key, []).append(value)

    def replace(self, key, value):
        ''' Replace the list of values with a single value. '''
        self.dict[key] = [value]

    def getall(self, key):
        ''' Return a (possibly empty) list of values for a key. '''
        return self.dict.get(key) or []

    #: Aliases for WTForms to mimic other multi-dict APIs (Django)
    getone = get
    getlist = getall


class FormsDict(MultiDict):
    ''' This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. '''

    #: Encoding used for attribute values.
    input_encoding = 'utf8'
    #: If true (default), unicode strings are first encoded with `latin1`
    #: and then decoded to match :attr:`input_encoding`.
    recode_unicode = True

    def _fix(self, s, encoding=None):
        if isinstance(s, unicode) and self.recode_unicode: # Python 3 WSGI
            return s.encode('latin1').decode(encoding or self.input_encoding)
        elif isinstance(s, bytes): # Python 2 WSGI
            return s.decode(encoding or self.input_encoding)
        else:
            return s

    def decode(self, encoding=None):
        ''' Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. '''
        copy = FormsDict()
        enc = copy.input_encoding = encoding or self.input_encoding
        copy.recode_unicode = False
        for key, value in self.allitems():
            copy.append(self._fix(key, enc), self._fix(value, enc))
        return copy

    def getunicode(self, name, default=None, encoding=None):
        ''' Return the value as a unicode string, or the default. '''
        try:
            return self._fix(self[name], encoding)
        except (UnicodeError, KeyError):
            return default

    def __getattr__(self, name, default=unicode()):
        # Without this guard, pickle generates a cryptic TypeError:
        if name.startswith('__') and name.endswith('__'):
            return super(FormsDict, self).__getattr__(name)
        return self.getunicode(name, default=default)


class HeaderDict(MultiDict):
    """ A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. """

    def __init__(self, *a, **ka):
        self.dict = {}
        if a or ka: self.update(*a, **ka)

    def __contains__(self, key): return _hkey(key) in self.dict
    def __delitem__(self, key): del self.dict[_hkey(key)]
    def __getitem__(self, key): return self.dict[_hkey(key)][-1]
    def __setitem__(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def append(self, key, value):
        self.dict.setdefault(_hkey(key), []).append(str(value))
    def replace(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def getall(self, key): return self.dict.get(_hkey(key)) or []
    def get(self, key, default=None, index=-1):
        return MultiDict.get(self, _hkey(key), default, index)
    def filter(self, names):
        for name in [_hkey(n) for n in names]:
            if name in self.dict:
                del self.dict[name]


class WSGIHeaderDict(DictMixin):
    ''' This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    '''
    #: List of keys that do not have a ``HTTP_`` prefix.
    cgikeys = ('CONTENT_TYPE', 'CONTENT_LENGTH')

    def __init__(self, environ):
        self.environ = environ

    def _ekey(self, key):
        ''' Translate header field name to CGI/WSGI environ key. '''
        key = key.replace('-','_').upper()
        if key in self.cgikeys:
            return key
        return 'HTTP_' + key

    def raw(self, key, default=None):
        ''' Return the header value as is (may be bytes or unicode). '''
        return self.environ.get(self._ekey(key), default)

    def __getitem__(self, key):
        return tonat(self.environ[self._ekey(key)], 'latin1')

    def __setitem__(self, key, value):
        raise TypeError("%s is read-only." % self.__class__)

    def __delitem__(self, key):
        raise TypeError("%s is read-only." % self.__class__)

    def __iter__(self):
        for key in self.environ:
            if key[:5] == 'HTTP_':
                yield key[5:].replace('_', '-').title()
            elif key in self.cgikeys:
                yield key.replace('_', '-').title()

    def keys(self): return [x for x in self]
    def __len__(self): return len(self.keys())
    def __contains__(self, key): return self._ekey(key) in self.environ



class ConfigDict(dict):
    ''' A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    '''
    __slots__ = ('_meta', '_on_change')

    class Namespace(DictMixin):

        def __init__(self, config, namespace):
            self._config = config
            self._prefix = namespace

        def __getitem__(self, key):
            depr('Accessing namespaces as dicts is discouraged. '
                 'Only use flat item access: '
                 'cfg["names"]["pace"]["key"] -> cfg["name.space.key"]') #0.12
            return self._config[self._prefix + '.' + key]

        def __setitem__(self, key, value):
            self._config[self._prefix + '.' + key] = value

        def __delitem__(self, key):
            del self._config[self._prefix + '.' + key]

        def __iter__(self):
            ns_prefix = self._prefix + '.'
            for key in self._config:
                ns, dot, name = key.rpartition('.')
                if ns == self._prefix and name:
                    yield name

        def keys(self): return [x for x in self]
        def __len__(self): return len(self.keys())
        def __contains__(self, key): return self._prefix + '.' + key in self._config
        def __repr__(self): return '<Config.Namespace %s.*>' % self._prefix
        def __str__(self): return '<Config.Namespace %s.*>' % self._prefix

        # Deprecated ConfigDict features
        def __getattr__(self, key):
            depr('Attribute access is deprecated.') #0.12
            if key not in self and key[0].isupper():
                self[key] = ConfigDict.Namespace(self._config, self._prefix + '.' + key)
            if key not in self and key.startswith('__'):
                raise AttributeError(key)
            return self.get(key)

        def __setattr__(self, key, value):
            if key in ('_config', '_prefix'):
                self.__dict__[key] = value
                return
            depr('Attribute assignment is deprecated.') #0.12
            if hasattr(DictMixin, key):
                raise AttributeError('Read-only attribute.')
            if key in self and self[key] and isinstance(self[key], self.__class__):
                raise AttributeError('Non-empty namespace attribute.')
            self[key] = value

        def __delattr__(self, key):
            if key in self:
                val = self.pop(key)
                if isinstance(val, self.__class__):
                    prefix = key + '.'
                    for key in self:
                        if key.startswith(prefix):
                            del self[prefix+key]

        def __call__(self, *a, **ka):
            depr('Calling ConfDict is deprecated. Use the update() method.') #0.12
            self.update(*a, **ka)
            return self

    def __init__(self, *a, **ka):
        self._meta = {}
        self._on_change = lambda name, value: None
        if a or ka:
            depr('Constructor does no longer accept parameters.') #0.12
            self.update(*a, **ka)

    def load_config(self, filename):
        ''' Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        '''
        conf = ConfigParser()
        conf.read(filename)
        for section in conf.sections():
            for key, value in conf.items(section):
                if section not in ('DEFAULT', 'bottle'):
                    key = section + '.' + key
                self[key] = value
        return self

    def load_dict(self, source, namespace='', make_namespaces=False):
        ''' Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        '''
        stack = [(namespace, source)]
        while stack:
            prefix, source = stack.pop()
            if not isinstance(source, dict):
                raise TypeError('Source is not a dict (r)' % type(key))
            for key, value in source.items():
                if not isinstance(key, basestring):
                    raise TypeError('Key is not a string (%r)' % type(key))
                full_key = prefix + '.' + key if prefix else key
                if isinstance(value, dict):
                    stack.append((full_key, value))
                    if make_namespaces:
                        self[full_key] = self.Namespace(self, full_key)
                else:
                    self[full_key] = value
        return self

    def update(self, *a, **ka):
        ''' If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` '''
        prefix = ''
        if a and isinstance(a[0], basestring):
            prefix = a[0].strip('.') + '.'
            a = a[1:]
        for key, value in dict(*a, **ka).items():
            self[prefix+key] = value

    def setdefault(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]

    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise TypeError('Key has type %r (not a string)' % type(key))

        value = self.meta_get(key, 'filter', lambda x: x)(value)
        if key in self and self[key] is value:
            return
        self._on_change(key, value)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)

    def clear(self):
        for key in self:
            del self[key]

    def meta_get(self, key, metafield, default=None):
        ''' Return the value of a meta field for a key. '''
        return self._meta.get(key, {}).get(metafield, default)

    def meta_set(self, key, metafield, value):
        ''' Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. '''
        self._meta.setdefault(key, {})[metafield] = value
        if key in self:
            self[key] = self[key]

    def meta_list(self, key):
        ''' Return an iterable of meta field names defined for a key. '''
        return self._meta.get(key, {}).keys()

    # Deprecated ConfigDict features
    def __getattr__(self, key):
        depr('Attribute access is deprecated.') #0.12
        if key not in self and key[0].isupper():
            self[key] = self.Namespace(self, key)
        if key not in self and key.startswith('__'):
            raise AttributeError(key)
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__slots__:
            return dict.__setattr__(self, key, value)
        depr('Attribute assignment is deprecated.') #0.12
        if hasattr(dict, key):
            raise AttributeError('Read-only attribute.')
        if key in self and self[key] and isinstance(self[key], self.Namespace):
            raise AttributeError('Non-empty namespace attribute.')
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            val = self.pop(key)
            if isinstance(val, self.Namespace):
                prefix = key + '.'
                for key in self:
                    if key.startswith(prefix):
                        del self[prefix+key]

    def __call__(self, *a, **ka):
        depr('Calling ConfDict is deprecated. Use the update() method.') #0.12
        self.update(*a, **ka)
        return self



class AppStack(list):
    """ A stack-like list. Calling it returns the head of the stack. """

    def __call__(self):
        """ Return the current default application. """
        return self[-1]

    def push(self, value=None):
        """ Add a new :class:`Bottle` instance to the stack """
        if not isinstance(value, Bottle):
            value = Bottle()
        self.append(value)
        return value


class WSGIFileWrapper(object):

    def __init__(self, fp, buffer_size=1024*64):
        self.fp, self.buffer_size = fp, buffer_size
        for attr in ('fileno', 'close', 'read', 'readlines', 'tell', 'seek'):
            if hasattr(fp, attr): setattr(self, attr, getattr(fp, attr))

    def __iter__(self):
        buff, read = self.buffer_size, self.read
        while True:
            part = read(buff)
            if not part: return
            yield part


class _closeiter(object):
    ''' This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). '''

    def __init__(self, iterator, close=None):
        self.iterator = iterator
        self.close_callbacks = makelist(close)

    def __iter__(self):
        return iter(self.iterator)

    def close(self):
        for func in self.close_callbacks:
            func()


class ResourceManager(object):
    ''' This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    '''

    def __init__(self, base='./', opener=open, cachemode='all'):
        self.opener = open
        self.base = base
        self.cachemode = cachemode

        #: A list of search paths. See :meth:`add_path` for details.
        self.path = []
        #: A cache for resolved paths. ``res.cache.clear()`` clears the cache.
        self.cache = {}

    def add_path(self, path, base=None, index=None, create=False):
        ''' Add a new path to the list of search paths. Return False if the
            path does not exist.

            :param path: The new search path. Relative paths are turned into
                an absolute and normalized form. If the path looks like a file
                (not ending in `/`), the filename is stripped off.
            :param base: Path used to absolutize relative search paths.
                Defaults to :attr:`base` which defaults to ``os.getcwd()``.
            :param index: Position within the list of search paths. Defaults
                to last index (appends to the list).

            The `base` parameter makes it easy to reference files installed
            along with a python module or package::

                res.add_path('./resources/', __file__)
        '''
        base = os.path.abspath(os.path.dirname(base or self.base))
        path = os.path.abspath(os.path.join(base, os.path.dirname(path)))
        path += os.sep
        if path in self.path:
            self.path.remove(path)
        if create and not os.path.isdir(path):
            os.makedirs(path)
        if index is None:
            self.path.append(path)
        else:
            self.path.insert(index, path)
        self.cache.clear()
        return os.path.exists(path)

    def __iter__(self):
        ''' Iterate over all existing files in all registered paths. '''
        search = self.path[:]
        while search:
            path = search.pop()
            if not os.path.isdir(path): continue
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isdir(full): search.append(full)
                else: yield full

    def lookup(self, name):
        ''' Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. '''
        if name not in self.cache or DEBUG:
            for path in self.path:
                fpath = os.path.join(path, name)
                if os.path.isfile(fpath):
                    if self.cachemode in ('all', 'found'):
                        self.cache[name] = fpath
                    return fpath
            if self.cachemode == 'all':
                self.cache[name] = None
        return self.cache[name]

    def open(self, name, mode='r', *args, **kwargs):
        ''' Find a resource and return a file object, or raise IOError. '''
        fname = self.lookup(name)
        if not fname: raise IOError("Resource %r not found." % name)
        return self.opener(fname, mode=mode, *args, **kwargs)


class FileUpload(object):

    def __init__(self, fileobj, name, filename, headers=None):
        ''' Wrapper for file uploads. '''
        #: Open file(-like) object (BytesIO buffer or temporary file)
        self.file = fileobj
        #: Name of the upload form field
        self.name = name
        #: Raw filename as sent by the client (may contain unsafe characters)
        self.raw_filename = filename
        #: A :class:`HeaderDict` with additional headers (e.g. content-type)
        self.headers = HeaderDict(headers) if headers else HeaderDict()

    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int, default=-1)

    @cached_property
    def filename(self):
        ''' Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        '''
        fname = self.raw_filename
        if not isinstance(fname, unicode):
            fname = fname.decode('utf8', 'ignore')
        fname = normalize('NFKD', fname).encode('ASCII', 'ignore').decode('ASCII')
        fname = os.path.basename(fname.replace('\\', os.path.sep))
        fname = re.sub(r'[^a-zA-Z0-9-_.\s]', '', fname).strip()
        fname = re.sub(r'[-\s]+', '-', fname).strip('.-')
        return fname[:255] or 'empty'

    def _copy_file(self, fp, chunk_size=2**16):
        read, write, offset = self.file.read, fp.write, self.file.tell()
        while 1:
            buf = read(chunk_size)
            if not buf: break
            write(buf)
        self.file.seek(offset)

    def save(self, destination, overwrite=False, chunk_size=2**16):
        ''' Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        '''
        if isinstance(destination, basestring): # Except file-likes here
            if os.path.isdir(destination):
                destination = os.path.join(destination, self.filename)
            if not overwrite and os.path.exists(destination):
                raise IOError('File exists.')
            with open(destination, 'wb') as fp:
                self._copy_file(fp, chunk_size)
        else:
            self._copy_file(destination, chunk_size)






###############################################################################
# Application Helper ###########################################################
###############################################################################


def abort(code=500, text='Unknown Error.'):
    """ Aborts execution and causes a HTTP error. """
    raise HTTPError(code, text)


def redirect(url, code=None):
    """ Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. """
    if not code:
        code = 303 if request.get('SERVER_PROTOCOL') == "HTTP/1.1" else 302
    res = response.copy(cls=HTTPResponse)
    res.status = code
    res.body = ""
    res.set_header('Location', urljoin(request.url, url))
    raise res


def _file_iter_range(fp, offset, bytes, maxread=1024*1024):
    ''' Yield chunks from a range in a file. No chunk is bigger than maxread.'''
    fp.seek(offset)
    while bytes > 0:
        part = fp.read(min(bytes, maxread))
        if not part: break
        bytes -= len(part)
        yield part


def static_file(filename, root, mimetype='auto', download=False, charset='UTF-8'):
    """ Open a file in a safe way and return :exc:`HTTPResponse` with status
        code 200, 305, 403 or 404. The ``Content-Type``, ``Content-Encoding``,
        ``Content-Length`` and ``Last-Modified`` headers are set if possible.
        Special support for ``If-Modified-Since``, ``Range`` and ``HEAD``
        requests.

        :param filename: Name or path of the file to send.
        :param root: Root path for file lookups. Should be an absolute directory
            path.
        :param mimetype: Defines the content-type header (default: guess from
            file extension)
        :param download: If True, ask the browser to open a `Save as...` dialog
            instead of opening the file with the associated program. You can
            specify a custom filename as a string. If not specified, the
            original filename is used (default: False).
        :param charset: The charset to use for files with a ``text/*``
            mime-type. (default: UTF-8)
    """

    root = os.path.abspath(root) + os.sep
    filename = os.path.abspath(os.path.join(root, filename.strip('/\\')))
    headers = dict()

    if not filename.startswith(root):
        return HTTPError(403, "Access denied.")
    if not os.path.exists(filename) or not os.path.isfile(filename):
        return HTTPError(404, "File does not exist.")
    if not os.access(filename, os.R_OK):
        return HTTPError(403, "You do not have permission to access this file.")

    if mimetype == 'auto':
        mimetype, encoding = mimetypes.guess_type(filename)
        if encoding: headers['Content-Encoding'] = encoding

    if mimetype:
        if mimetype[:5] == 'text/' and charset and 'charset' not in mimetype:
            mimetype += '; charset=%s' % charset
        headers['Content-Type'] = mimetype

    if download:
        download = os.path.basename(filename if download == True else download)
        headers['Content-Disposition'] = 'attachment; filename="%s"' % download

    stats = os.stat(filename)
    headers['Content-Length'] = clen = stats.st_size
    lm = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stats.st_mtime))
    headers['Last-Modified'] = lm

    ims = request.environ.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims = parse_date(ims.split(";")[0].strip())
    if ims is not None and ims >= int(stats.st_mtime):
        headers['Date'] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        return HTTPResponse(status=304, **headers)

    body = '' if request.method == 'HEAD' else open(filename, 'rb')

    headers["Accept-Ranges"] = "bytes"
    ranges = request.environ.get('HTTP_RANGE')
    if 'HTTP_RANGE' in request.environ:
        ranges = list(parse_range_header(request.environ['HTTP_RANGE'], clen))
        if not ranges:
            return HTTPError(416, "Requested Range Not Satisfiable")
        offset, end = ranges[0]
        headers["Content-Range"] = "bytes %d-%d/%d" % (offset, end-1, clen)
        headers["Content-Length"] = str(end-offset)
        if body: body = _file_iter_range(body, offset, end-offset)
        return HTTPResponse(body, status=206, **headers)
    return HTTPResponse(body, **headers)






###############################################################################
# HTTP Utilities and MISC (TODO) ###############################################
###############################################################################


def debug(mode=True):
    """ Change the debug level.
    There is only one debug level supported at the moment."""
    global DEBUG
    if mode: warnings.simplefilter('default')
    DEBUG = bool(mode)

def http_date(value):
    if isinstance(value, (datedate, datetime)):
        value = value.utctimetuple()
    elif isinstance(value, (int, float)):
        value = time.gmtime(value)
    if not isinstance(value, basestring):
        value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
    return value

def parse_date(ims):
    """ Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. """
    try:
        ts = email.utils.parsedate_tz(ims)
        return time.mktime(ts[:8] + (0,)) - (ts[9] or 0) - time.timezone
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def parse_auth(header):
    """ Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or None"""
    try:
        method, data = header.split(None, 1)
        if method.lower() == 'basic':
            user, pwd = touni(base64.b64decode(tob(data))).split(':',1)
            return user, pwd
    except (KeyError, ValueError):
        return None

def parse_range_header(header, maxlen=0):
    ''' Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.'''
    if not header or header[:6] != 'bytes=': return
    ranges = [r.split('-', 1) for r in header[6:].split(',') if '-' in r]
    for start, end in ranges:
        try:
            if not start:  # bytes=-100    -> last 100 bytes
                start, end = max(0, maxlen-int(end)), maxlen
            elif not end:  # bytes=100-    -> all but the first 99 bytes
                start, end = int(start), maxlen
            else:          # bytes=100-200 -> bytes 100-200 (inclusive)
                start, end = int(start), min(int(end)+1, maxlen)
            if 0 <= start < end <= maxlen:
                yield start, end
        except ValueError:
            pass

def _parse_qsl(qs):
    r = []
    for pair in qs.replace(';','&').split('&'):
        if not pair: continue
        nv = pair.split('=', 1)
        if len(nv) != 2: nv.append('')
        key = urlunquote(nv[0].replace('+', ' '))
        value = urlunquote(nv[1].replace('+', ' '))
        r.append((key, value))
    return r

def _lscmp(a, b):
    ''' Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. '''
    return not sum(0 if x==y else 1 for x, y in zip(a, b)) and len(a) == len(b)


def cookie_encode(data, key):
    ''' Encode and sign a pickle-able object. Return a (byte) string '''
    msg = base64.b64encode(pickle.dumps(data, -1))
    sig = base64.b64encode(hmac.new(tob(key), msg).digest())
    return tob('!') + sig + tob('?') + msg


def cookie_decode(data, key):
    ''' Verify and decode an encoded string. Return an object or None.'''
    data = tob(data)
    if cookie_is_encoded(data):
        sig, msg = data.split(tob('?'), 1)
        if _lscmp(sig[1:], base64.b64encode(hmac.new(tob(key), msg).digest())):
            return pickle.loads(base64.b64decode(msg))
    return None


def cookie_is_encoded(data):
    ''' Return True if the argument looks like a encoded cookie.'''
    return bool(data.startswith(tob('!')) and tob('?') in data)


def html_escape(string):
    ''' Escape HTML special characters ``&<>`` and quotes ``'"``. '''
    return string.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')\
                 .replace('"','&quot;').replace("'",'&#039;')


def html_quote(string):
    ''' Escape and quote a string to be used as an HTTP attribute.'''
    return '"%s"' % html_escape(string).replace('\n','&#10;')\
                    .replace('\r','&#13;').replace('\t','&#9;')


def yieldroutes(func):
    """ Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    """
    path = '/' + func.__name__.replace('__','/').lstrip('/')
    spec = getargspec(func)
    argc = len(spec[0]) - len(spec[3] or [])
    path += ('/<%s>' * argc) % tuple(spec[0][:argc])
    yield path
    for arg in spec[0][argc:]:
        path += '/<%s>' % arg
        yield path


def path_shift(script_name, path_info, shift=1):
    ''' Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    '''
    if shift == 0: return script_name, path_info
    pathlist = path_info.strip('/').split('/')
    scriptlist = script_name.strip('/').split('/')
    if pathlist and pathlist[0] == '': pathlist = []
    if scriptlist and scriptlist[0] == '': scriptlist = []
    if shift > 0 and shift <= len(pathlist):
        moved = pathlist[:shift]
        scriptlist = scriptlist + moved
        pathlist = pathlist[shift:]
    elif shift < 0 and shift >= -len(scriptlist):
        moved = scriptlist[shift:]
        pathlist = moved + pathlist
        scriptlist = scriptlist[:shift]
    else:
        empty = 'SCRIPT_NAME' if shift < 0 else 'PATH_INFO'
        raise AssertionError("Cannot shift. Nothing left from %s" % empty)
    new_script_name = '/' + '/'.join(scriptlist)
    new_path_info = '/' + '/'.join(pathlist)
    if path_info.endswith('/') and pathlist: new_path_info += '/'
    return new_script_name, new_path_info


def auth_basic(check, realm="private", text="Access denied"):
    ''' Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. '''
    def decorator(func):
        def wrapper(*a, **ka):
            user, password = request.auth or (None, None)
            if user is None or not check(user, password):
                err = HTTPError(401, text)
                err.add_header('WWW-Authenticate', 'Basic realm="%s"' % realm)
                return err
            return func(*a, **ka)
        return wrapper
    return decorator


# Shortcuts for common Bottle methods.
# They all refer to the current default application.

def make_default_app_wrapper(name):
    ''' Return a callable that relays calls to the current default app. '''
    @functools.wraps(getattr(Bottle, name))
    def wrapper(*a, **ka):
        return getattr(app(), name)(*a, **ka)
    return wrapper

route     = make_default_app_wrapper('route')
get       = make_default_app_wrapper('get')
post      = make_default_app_wrapper('post')
put       = make_default_app_wrapper('put')
delete    = make_default_app_wrapper('delete')
error     = make_default_app_wrapper('error')
mount     = make_default_app_wrapper('mount')
hook      = make_default_app_wrapper('hook')
install   = make_default_app_wrapper('install')
uninstall = make_default_app_wrapper('uninstall')
url       = make_default_app_wrapper('get_url')







###############################################################################
# Server Adapter ###############################################################
###############################################################################


class ServerAdapter(object):
    quiet = False
    def __init__(self, host='127.0.0.1', port=8080, **options):
        self.options = options
        self.host = host
        self.port = int(port)

    def run(self, handler): # pragma: no cover
        pass

    def __repr__(self):
        args = ', '.join(['%s=%s'%(k,repr(v)) for k, v in self.options.items()])
        return "%s(%s)" % (self.__class__.__name__, args)


class CGIServer(ServerAdapter):
    quiet = True
    def run(self, handler): # pragma: no cover
        from wsgiref.handlers import CGIHandler
        def fixed_environ(environ, start_response):
            environ.setdefault('PATH_INFO', '')
            return handler(environ, start_response)
        CGIHandler().run(fixed_environ)


class FlupFCGIServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        import flup.server.fcgi
        self.options.setdefault('bindAddress', (self.host, self.port))
        flup.server.fcgi.WSGIServer(handler, **self.options).run()


class WSGIRefServer(ServerAdapter):
    def run(self, app): # pragma: no cover
        from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
        from wsgiref.simple_server import make_server
        import socket

        class FixedHandler(WSGIRequestHandler):
            def address_string(self): # Prevent reverse DNS lookups please.
                return self.client_address[0]
            def log_request(*args, **kw):
                if not self.quiet:
                    return WSGIRequestHandler.log_request(*args, **kw)

        handler_cls = self.options.get('handler_class', FixedHandler)
        server_cls  = self.options.get('server_class', WSGIServer)

        if ':' in self.host: # Fix wsgiref for IPv6 addresses.
            if getattr(server_cls, 'address_family') == socket.AF_INET:
                class server_cls(server_cls):
                    address_family = socket.AF_INET6

        srv = make_server(self.host, self.port, app, server_cls, handler_cls)
        srv.serve_forever()


class CherryPyServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        from cherrypy import wsgiserver
        self.options['bind_addr'] = (self.host, self.port)
        self.options['wsgi_app'] = handler

        certfile = self.options.get('certfile')
        if certfile:
            del self.options['certfile']
        keyfile = self.options.get('keyfile')
        if keyfile:
            del self.options['keyfile']

        server = wsgiserver.CherryPyWSGIServer(**self.options)
        if certfile:
            server.ssl_certificate = certfile
        if keyfile:
            server.ssl_private_key = keyfile

        try:
            server.start()
        finally:
            server.stop()


class WaitressServer(ServerAdapter):
    def run(self, handler):
        from waitress import serve
        serve(handler, host=self.host, port=self.port)


class PasteServer(ServerAdapter):
    def run(self, handler): # pragma: no cover
        from paste import httpserver
        from paste.translogger import TransLogger
        handler = TransLogger(handler, setup_console_handler=(not self.quiet))
        httpserver.serve(handler, host=self.host, port=str(self.port),
                         **self.options)


class MeinheldServer(ServerAdapter):
    def run(self, handler):
        from meinheld import server
        server.listen((self.host, self.port))
        server.run(handler)


class FapwsServer(ServerAdapter):
    """ Extremely fast webserver using libev. See http://www.fapws.org/ """
    def run(self, handler): # pragma: no cover
        import fapws._evwsgi as evwsgi
        from fapws import base, config
        port = self.port
        if float(config.SERVER_IDENT[-2:]) > 0.4:
            # fapws3 silently changed its API in 0.5
            port = str(port)
        evwsgi.start(self.host, port)
        # fapws3 never releases the GIL. Complain upstream. I tried. No luck.
        if 'BOTTLE_CHILD' in os.environ and not self.quiet:
            _stderr("WARNING: Auto-reloading does not work with Fapws3.\n")
            _stderr("         (Fapws3 breaks python thread support)\n")
        evwsgi.set_base_module(base)
        def app(environ, start_response):
            environ['wsgi.multiprocess'] = False
            return handler(environ, start_response)
        evwsgi.wsgi_cb(('', app))
        evwsgi.run()


class TornadoServer(ServerAdapter):
    """ The super hyped asynchronous server by facebook. Untested. """
    def run(self, handler): # pragma: no cover
        import tornado.wsgi, tornado.httpserver, tornado.ioloop
        container = tornado.wsgi.WSGIContainer(handler)
        server = tornado.httpserver.HTTPServer(container)
        server.listen(port=self.port,address=self.host)
        tornado.ioloop.IOLoop.instance().start()


class AppEngineServer(ServerAdapter):
    """ Adapter for Google App Engine. """
    quiet = True
    def run(self, handler):
        from google.appengine.ext.webapp import util
        # A main() function in the handler script enables 'App Caching'.
        # Lets makes sure it is there. This _really_ improves performance.
        module = sys.modules.get('__main__')
        if module and not hasattr(module, 'main'):
            module.main = lambda: util.run_wsgi_app(handler)
        util.run_wsgi_app(handler)


class TwistedServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from twisted.web import server, wsgi
        from twisted.python.threadpool import ThreadPool
        from twisted.internet import reactor
        thread_pool = ThreadPool()
        thread_pool.start()
        reactor.addSystemEventTrigger('after', 'shutdown', thread_pool.stop)
        factory = server.Site(wsgi.WSGIResource(reactor, thread_pool, handler))
        reactor.listenTCP(self.port, factory, interface=self.host)
        reactor.run()


class DieselServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from diesel.protocols.wsgi import WSGIApplication
        app = WSGIApplication(handler, port=self.port)
        app.run()


class GeventServer(ServerAdapter):
    """ Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    """
    def run(self, handler):
        from gevent import wsgi, pywsgi, local
        if not isinstance(threading.local(), local.local):
            msg = "Bottle requires gevent.monkey.patch_all() (before import)"
            raise RuntimeError(msg)
        if not self.options.pop('fast', None): wsgi = pywsgi
        self.options['log'] = None if self.quiet else 'default'
        address = (self.host, self.port)
        server = wsgi.WSGIServer(address, handler, **self.options)
        if 'BOTTLE_CHILD' in os.environ:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: server.stop())
        server.serve_forever()


class GeventSocketIOServer(ServerAdapter):
    def run(self,handler):
        from socketio import server
        address = (self.host, self.port)
        server.SocketIOServer(address, handler, **self.options).serve_forever()


class GunicornServer(ServerAdapter):
    """ Untested. See http://gunicorn.org/configure.html for options. """
    def run(self, handler):
        from gunicorn.app.base import Application

        config = {'bind': "%s:%d" % (self.host, int(self.port))}
        config.update(self.options)

        class GunicornApplication(Application):
            def init(self, parser, opts, args):
                return config

            def load(self):
                return handler

        GunicornApplication().run()


class EventletServer(ServerAdapter):
    """ Untested """
    def run(self, handler):
        from eventlet import wsgi, listen
        try:
            wsgi.server(listen((self.host, self.port)), handler,
                        log_output=(not self.quiet))
        except TypeError:
            # Fallback, if we have old version of eventlet
            wsgi.server(listen((self.host, self.port)), handler)


class RocketServer(ServerAdapter):
    """ Untested. """
    def run(self, handler):
        from rocket import Rocket
        server = Rocket((self.host, self.port), 'wsgi', { 'wsgi_app' : handler })
        server.start()


class BjoernServer(ServerAdapter):
    """ Fast server written in C: https://github.com/jonashaag/bjoern """
    def run(self, handler):
        from bjoern import run
        run(handler, self.host, self.port)


class AutoServer(ServerAdapter):
    """ Untested. """
    adapters = [WaitressServer, PasteServer, TwistedServer, CherryPyServer, WSGIRefServer]
    def run(self, handler):
        for sa in self.adapters:
            try:
                return sa(self.host, self.port, **self.options).run(handler)
            except ImportError:
                pass

server_names = {
    'cgi': CGIServer,
    'flup': FlupFCGIServer,
    'wsgiref': WSGIRefServer,
    'waitress': WaitressServer,
    'cherrypy': CherryPyServer,
    'paste': PasteServer,
    'fapws3': FapwsServer,
    'tornado': TornadoServer,
    'gae': AppEngineServer,
    'twisted': TwistedServer,
    'diesel': DieselServer,
    'meinheld': MeinheldServer,
    'gunicorn': GunicornServer,
    'eventlet': EventletServer,
    'gevent': GeventServer,
    'geventSocketIO':GeventSocketIOServer,
    'rocket': RocketServer,
    'bjoern' : BjoernServer,
    'auto': AutoServer,
}






###############################################################################
# Application Control ##########################################################
###############################################################################


def load(target, **namespace):
    """ Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    """
    module, target = target.split(":", 1) if ':' in target else (target, None)
    if module not in sys.modules: __import__(module)
    if not target: return sys.modules[module]
    if target.isalnum(): return getattr(sys.modules[module], target)
    package_name = module.split('.')[0]
    namespace[package_name] = sys.modules[package_name]
    return eval('%s.%s' % (module, target), namespace)


def load_app(target):
    """ Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. """
    global NORUN; NORUN, nr_old = True, NORUN
    try:
        tmp = default_app.push() # Create a new "default application"
        rv = load(target) # Import the target module
        return rv if callable(rv) else tmp
    finally:
        default_app.remove(tmp) # Remove the temporary added default application
        NORUN = nr_old

_debug = debug
def run(app=None, server='wsgiref', host='127.0.0.1', port=8080,
        interval=1, reloader=False, quiet=False, plugins=None,
        debug=None, **kargs):
    """ Start a server instance. This method blocks until the server terminates.

        :param app: WSGI application or target string supported by
               :func:`load_app`. (default: :func:`default_app`)
        :param server: Server adapter to use. See :data:`server_names` keys
               for valid names or pass a :class:`ServerAdapter` subclass.
               (default: `wsgiref`)
        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on
               all interfaces including the external one. (default: 127.0.0.1)
        :param port: Server port to bind to. Values below 1024 require root
               privileges. (default: 8080)
        :param reloader: Start auto-reloading server? (default: False)
        :param interval: Auto-reloader interval in seconds (default: 1)
        :param quiet: Suppress output to stdout and stderr? (default: False)
        :param options: Options passed to the server adapter.
     """
    if NORUN: return
    if reloader and not os.environ.get('BOTTLE_CHILD'):
        try:
            lockfile = None
            fd, lockfile = tempfile.mkstemp(prefix='bottle.', suffix='.lock')
            os.close(fd) # We only need this file to exist. We never write to it
            while os.path.exists(lockfile):
                args = [sys.executable] + sys.argv
                environ = os.environ.copy()
                environ['BOTTLE_CHILD'] = 'true'
                environ['BOTTLE_LOCKFILE'] = lockfile
                p = subprocess.Popen(args, env=environ)
                while p.poll() is None: # Busy wait...
                    os.utime(lockfile, None) # I am alive!
                    time.sleep(interval)
                if p.poll() != 3:
                    if os.path.exists(lockfile): os.unlink(lockfile)
                    sys.exit(p.poll())
        except KeyboardInterrupt:
            pass
        finally:
            if os.path.exists(lockfile):
                os.unlink(lockfile)
        return

    try:
        if debug is not None: _debug(debug)
        app = app or default_app()
        if isinstance(app, basestring):
            app = load_app(app)
        if not callable(app):
            raise ValueError("Application is not callable: %r" % app)

        for plugin in plugins or []:
            app.install(plugin)

        if server in server_names:
            server = server_names.get(server)
        if isinstance(server, basestring):
            server = load(server)
        if isinstance(server, type):
            server = server(host=host, port=port, **kargs)
        if not isinstance(server, ServerAdapter):
            raise ValueError("Unknown or unsupported server: %r" % server)

        server.quiet = server.quiet or quiet
        if not server.quiet:
            _stderr("Bottle v%s server starting up (using %s)...\n" % (__version__, repr(server)))
            _stderr("Listening on http://%s:%d/\n" % (server.host, server.port))
            _stderr("Hit Ctrl-C to quit.\n\n")

        if reloader:
            lockfile = os.environ.get('BOTTLE_LOCKFILE')
            bgcheck = FileCheckerThread(lockfile, interval)
            with bgcheck:
                server.run(app)
            if bgcheck.status == 'reload':
                sys.exit(3)
        else:
            server.run(app)
    except KeyboardInterrupt:
        pass
    except (SystemExit, MemoryError):
        raise
    except:
        if not reloader: raise
        if not getattr(server, 'quiet', quiet):
            print_exc()
        time.sleep(interval)
        sys.exit(3)



class FileCheckerThread(threading.Thread):
    ''' Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. '''

    def __init__(self, lockfile, interval):
        threading.Thread.__init__(self)
        self.lockfile, self.interval = lockfile, interval
        #: Is one of 'reload', 'error' or 'exit'
        self.status = None

    def run(self):
        exists = os.path.exists
        mtime = lambda path: os.stat(path).st_mtime
        files = dict()

        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', '')
            if path[-4:] in ('.pyo', '.pyc'): path = path[:-1]
            if path and exists(path): files[path] = mtime(path)

        while not self.status:
            if not exists(self.lockfile)\
            or mtime(self.lockfile) < time.time() - self.interval - 5:
                self.status = 'error'
                thread.interrupt_main()
            for path, lmtime in list(files.items()):
                if not exists(path) or mtime(path) > lmtime:
                    self.status = 'reload'
                    thread.interrupt_main()
                    break
            time.sleep(self.interval)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.status: self.status = 'exit' # silent exit
        self.join()
        return exc_type is not None and issubclass(exc_type, KeyboardInterrupt)





###############################################################################
# Template Adapters ############################################################
###############################################################################


class TemplateError(HTTPError):
    def __init__(self, message):
        HTTPError.__init__(self, 500, message)


class BaseTemplate(object):
    """ Base class and minimal API for template adapters """
    extensions = ['tpl','html','thtml','stpl']
    settings = {} #used in prepare()
    defaults = {} #used in render()

    def __init__(self, source=None, name=None, lookup=[], encoding='utf8', **settings):
        """ Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        """
        self.name = name
        self.source = source.read() if hasattr(source, 'read') else source
        self.filename = source.filename if hasattr(source, 'filename') else None
        self.lookup = [os.path.abspath(x) for x in lookup]
        self.encoding = encoding
        self.settings = self.settings.copy() # Copy from class variable
        self.settings.update(settings) # Apply
        if not self.source and self.name:
            self.filename = self.search(self.name, self.lookup)
            if not self.filename:
                raise TemplateError('Template %s not found.' % repr(name))
        if not self.source and not self.filename:
            raise TemplateError('No template specified.')
        self.prepare(**self.settings)

    @classmethod
    def search(cls, name, lookup=[]):
        """ Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. """
        if not lookup:
            depr('The template lookup path list should not be empty.') #0.12
            lookup = ['.']

        if os.path.isabs(name) and os.path.isfile(name):
            depr('Absolute template path names are deprecated.') #0.12
            return os.path.abspath(name)

        for spath in lookup:
            spath = os.path.abspath(spath) + os.sep
            fname = os.path.abspath(os.path.join(spath, name))
            if not fname.startswith(spath): continue
            if os.path.isfile(fname): return fname
            for ext in cls.extensions:
                if os.path.isfile('%s.%s' % (fname, ext)):
                    return '%s.%s' % (fname, ext)

    @classmethod
    def global_config(cls, key, *args):
        ''' This reads or sets the global settings stored in class.settings. '''
        if args:
            cls.settings = cls.settings.copy() # Make settings local to class
            cls.settings[key] = args[0]
        else:
            return cls.settings[key]

    def prepare(self, **options):
        """ Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        """
        raise NotImplementedError

    def render(self, *args, **kwargs):
        """ Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        """
        raise NotImplementedError


class MakoTemplate(BaseTemplate):
    def prepare(self, **options):
        from mako.template import Template
        from mako.lookup import TemplateLookup
        options.update({'input_encoding':self.encoding})
        options.setdefault('format_exceptions', bool(DEBUG))
        lookup = TemplateLookup(directories=self.lookup, **options)
        if self.source:
            self.tpl = Template(self.source, lookup=lookup, **options)
        else:
            self.tpl = Template(uri=self.name, filename=self.filename, lookup=lookup, **options)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)


class CheetahTemplate(BaseTemplate):
    def prepare(self, **options):
        from Cheetah.Template import Template
        self.context = threading.local()
        self.context.vars = {}
        options['searchList'] = [self.context.vars]
        if self.source:
            self.tpl = Template(source=self.source, **options)
        else:
            self.tpl = Template(file=self.filename, **options)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        self.context.vars.update(self.defaults)
        self.context.vars.update(kwargs)
        out = str(self.tpl)
        self.context.vars.clear()
        return out


class Jinja2Template(BaseTemplate):
    def prepare(self, filters=None, tests=None, globals={}, **kwargs):
        from jinja2 import Environment, FunctionLoader
        if 'prefix' in kwargs: # TODO: to be removed after a while
            raise RuntimeError('The keyword argument `prefix` has been removed. '
                'Use the full jinja2 environment name line_statement_prefix instead.')
        self.env = Environment(loader=FunctionLoader(self.loader), **kwargs)
        if filters: self.env.filters.update(filters)
        if tests: self.env.tests.update(tests)
        if globals: self.env.globals.update(globals)
        if self.source:
            self.tpl = self.env.from_string(self.source)
        else:
            self.tpl = self.env.get_template(self.filename)

    def render(self, *args, **kwargs):
        for dictarg in args: kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)

    def loader(self, name):
        fname = self.search(name, self.lookup)
        if not fname: return
        with open(fname, "rb") as f:
            return f.read().decode(self.encoding)


class SimpleTemplate(BaseTemplate):

    def prepare(self, escape_func=html_escape, noescape=False, syntax=None, **ka):
        self.cache = {}
        enc = self.encoding
        self._str = lambda x: touni(x, enc)
        self._escape = lambda x: escape_func(touni(x, enc))
        self.syntax = syntax
        if noescape:
            self._str, self._escape = self._escape, self._str

    @cached_property
    def co(self):
        return compile(self.code, self.filename or '<string>', 'exec')

    @cached_property
    def code(self):
        source = self.source
        if not source:
            with open(self.filename, 'rb') as f:
                source = f.read()
        try:
            source, encoding = touni(source), 'utf8'
        except UnicodeError:
            depr('Template encodings other than utf8 are no longer supported.') #0.11
            source, encoding = touni(source, 'latin1'), 'latin1'
        parser = StplParser(source, encoding=encoding, syntax=self.syntax)
        code = parser.translate()
        self.encoding = parser.encoding
        return code

    def _rebase(self, _env, _name=None, **kwargs):
        if _name is None:
            depr('Rebase function called without arguments.'
                 ' You were probably looking for {{base}}?', True) #0.12
        _env['_rebase'] = (_name, kwargs)

    def _include(self, _env, _name=None, **kwargs):
        if _name is None:
            depr('Rebase function called without arguments.'
                 ' You were probably looking for {{base}}?', True) #0.12
        env = _env.copy()
        env.update(kwargs)
        if _name not in self.cache:
            self.cache[_name] = self.__class__(name=_name, lookup=self.lookup)
        return self.cache[_name].execute(env['_stdout'], env)

    def execute(self, _stdout, kwargs):
        env = self.defaults.copy()
        env.update(kwargs)
        env.update({'_stdout': _stdout, '_printlist': _stdout.extend,
            'include': functools.partial(self._include, env),
            'rebase': functools.partial(self._rebase, env), '_rebase': None,
            '_str': self._str, '_escape': self._escape, 'get': env.get,
            'setdefault': env.setdefault, 'defined': env.__contains__ })
        eval(self.co, env)
        if env.get('_rebase'):
            subtpl, rargs = env.pop('_rebase')
            rargs['base'] = ''.join(_stdout) #copy stdout
            del _stdout[:] # clear stdout
            return self._include(env, subtpl, **rargs)
        return env

    def render(self, *args, **kwargs):
        """ Render the template using keyword arguments as local variables. """
        env = {}; stdout = []
        for dictarg in args: env.update(dictarg)
        env.update(kwargs)
        self.execute(stdout, env)
        return ''.join(stdout)


class StplSyntaxError(TemplateError): pass


class StplParser(object):
    ''' Parser for stpl templates. '''
    _re_cache = {} #: Cache for compiled re patterns
    # This huge pile of voodoo magic splits python code into 8 different tokens.
    # 1: All kinds of python strings (trust me, it works)
    _re_tok = '((?m)[urbURB]?(?:\'\'(?!\')|""(?!")|\'{6}|"{6}' \
               '|\'(?:[^\\\\\']|\\\\.)+?\'|"(?:[^\\\\"]|\\\\.)+?"' \
               '|\'{3}(?:[^\\\\]|\\\\.|\\n)+?\'{3}' \
               '|"{3}(?:[^\\\\]|\\\\.|\\n)+?"{3}))'
    _re_inl = _re_tok.replace('|\\n','') # We re-use this string pattern later
    # 2: Comments (until end of line, but not the newline itself)
    _re_tok += '|(#.*)'
    # 3,4: Open and close grouping tokens
    _re_tok += '|([\[\{\(])'
    _re_tok += '|([\]\}\)])'
    # 5,6: Keywords that start or continue a python block (only start of line)
    _re_tok += '|^([ \\t]*(?:if|for|while|with|try|def|class)\\b)' \
               '|^([ \\t]*(?:elif|else|except|finally)\\b)'
    # 7: Our special 'end' keyword (but only if it stands alone)
    _re_tok += '|((?:^|;)[ \\t]*end[ \\t]*(?=(?:%(block_close)s[ \\t]*)?\\r?$|;|#))'
    # 8: A customizable end-of-code-block template token (only end of line)
    _re_tok += '|(%(block_close)s[ \\t]*(?=$))'
    # 9: And finally, a single newline. The 10th token is 'everything else'
    _re_tok += '|(\\r?\\n)'

    # Match the start tokens of code areas in a template
    _re_split = '(?m)^[ \t]*(\\\\?)((%(line_start)s)|(%(block_start)s))(%%?)'
    # Match inline statements (may contain python strings)
    _re_inl = '%%(inline_start)s((?:%s|[^\'"\n]*?)+)%%(inline_end)s' % _re_inl

    default_syntax = '<% %> % {{ }}'

    def __init__(self, source, syntax=None, encoding='utf8'):
        self.source, self.encoding = touni(source, encoding), encoding
        self.set_syntax(syntax or self.default_syntax)
        self.code_buffer, self.text_buffer = [], []
        self.lineno, self.offset = 1, 0
        self.indent, self.indent_mod = 0, 0
        self.paren_depth = 0

    def get_syntax(self):
        ''' Tokens as a space separated string (default: <% %> % {{ }}) '''
        return self._syntax

    def set_syntax(self, syntax):
        self._syntax = syntax
        self._tokens = syntax.split()
        if not syntax in self._re_cache:
            names = 'block_start block_close line_start inline_start inline_end'
            etokens = map(re.escape, self._tokens)
            pattern_vars = dict(zip(names.split(), etokens))
            patterns = (self._re_split, self._re_tok, self._re_inl)
            patterns = [re.compile(p%pattern_vars) for p in patterns]
            self._re_cache[syntax] = patterns
        self.re_split, self.re_tok, self.re_inl = self._re_cache[syntax]

    syntax = property(get_syntax, set_syntax)

    def translate(self):
        if self.offset: raise RuntimeError('Parser is a one time instance.')
        while True:
            m = self.re_split.search(self.source[self.offset:])
            if m:
                text = self.source[self.offset:self.offset+m.start()]
                self.text_buffer.append(text)
                self.offset += m.end()
                if m.group(1): # New escape syntax
                    line, sep, _ = self.source[self.offset:].partition('\n')
                    self.text_buffer.append(m.group(2)+m.group(5)+line+sep)
                    self.offset += len(line+sep)+1
                    continue
                elif m.group(5): # Old escape syntax
                    depr('Escape code lines with a backslash.') #0.12
                    line, sep, _ = self.source[self.offset:].partition('\n')
                    self.text_buffer.append(m.group(2)+line+sep)
                    self.offset += len(line+sep)+1
                    continue
                self.flush_text()
                self.read_code(multiline=bool(m.group(4)))
            else: break
        self.text_buffer.append(self.source[self.offset:])
        self.flush_text()
        return ''.join(self.code_buffer)

    def read_code(self, multiline):
        code_line, comment = '', ''
        while True:
            m = self.re_tok.search(self.source[self.offset:])
            if not m:
                code_line += self.source[self.offset:]
                self.offset = len(self.source)
                self.write_code(code_line.strip(), comment)
                return
            code_line += self.source[self.offset:self.offset+m.start()]
            self.offset += m.end()
            _str, _com, _po, _pc, _blk1, _blk2, _end, _cend, _nl = m.groups()
            if (code_line or self.paren_depth > 0) and (_blk1 or _blk2): # a if b else c
                code_line += _blk1 or _blk2
                continue
            if _str:    # Python string
                code_line += _str
            elif _com:  # Python comment (up to EOL)
                comment = _com
                if multiline and _com.strip().endswith(self._tokens[1]):
                    multiline = False # Allow end-of-block in comments
            elif _po:  # open parenthesis
                self.paren_depth += 1
                code_line += _po
            elif _pc:  # close parenthesis
                if self.paren_depth > 0:
                    # we could check for matching parentheses here, but it's
                    # easier to leave that to python - just check counts
                    self.paren_depth -= 1
                code_line += _pc
            elif _blk1: # Start-block keyword (if/for/while/def/try/...)
                code_line, self.indent_mod = _blk1, -1
                self.indent += 1
            elif _blk2: # Continue-block keyword (else/elif/except/...)
                code_line, self.indent_mod = _blk2, -1
            elif _end:  # The non-standard 'end'-keyword (ends a block)
                self.indent -= 1
            elif _cend: # The end-code-block template token (usually '%>')
                if multiline: multiline = False
                else: code_line += _cend
            else: # \n
                self.write_code(code_line.strip(), comment)
                self.lineno += 1
                code_line, comment, self.indent_mod = '', '', 0
                if not multiline:
                    break

    def flush_text(self):
        text = ''.join(self.text_buffer)
        del self.text_buffer[:]
        if not text: return
        parts, pos, nl = [], 0, '\\\n'+'  '*self.indent
        for m in self.re_inl.finditer(text):
            prefix, pos = text[pos:m.start()], m.end()
            if prefix:
                parts.append(nl.join(map(repr, prefix.splitlines(True))))
            if prefix.endswith('\n'): parts[-1] += nl
            parts.append(self.process_inline(m.group(1).strip()))
        if pos < len(text):
            prefix = text[pos:]
            lines = prefix.splitlines(True)
            if lines[-1].endswith('\\\\\n'): lines[-1] = lines[-1][:-3]
            elif lines[-1].endswith('\\\\\r\n'): lines[-1] = lines[-1][:-4]
            parts.append(nl.join(map(repr, lines)))
        code = '_printlist((%s,))' % ', '.join(parts)
        self.lineno += code.count('\n')+1
        self.write_code(code)

    def process_inline(self, chunk):
        if chunk[0] == '!': return '_str(%s)' % chunk[1:]
        return '_escape(%s)' % chunk

    def write_code(self, line, comment=''):
        line, comment = self.fix_backward_compatibility(line, comment)
        code  = '  ' * (self.indent+self.indent_mod)
        code += line.lstrip() + comment + '\n'
        self.code_buffer.append(code)

    def fix_backward_compatibility(self, line, comment):
        parts = line.strip().split(None, 2)
        if parts and parts[0] in ('include', 'rebase'):
            depr('The include and rebase keywords are functions now.') #0.12
            if len(parts) == 1:   return "_printlist([base])", comment
            elif len(parts) == 2: return "_=%s(%r)" % tuple(parts), comment
            else:                 return "_=%s(%r, %s)" % tuple(parts), comment
        if self.lineno <= 2 and not line.strip() and 'coding' in comment:
            m = re.match(r"#.*coding[:=]\s*([-\w.]+)", comment)
            if m:
                depr('PEP263 encoding strings in templates are deprecated.') #0.12
                enc = m.group(1)
                self.source = self.source.encode(self.encoding).decode(enc)
                self.encoding = enc
                return line, comment.replace('coding','coding*')
        return line, comment


def template(*args, **kwargs):
    '''
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    '''
    tpl = args[0] if args else None
    adapter = kwargs.pop('template_adapter', SimpleTemplate)
    lookup = kwargs.pop('template_lookup', TEMPLATE_PATH)
    tplid = (id(lookup), tpl)
    if tplid not in TEMPLATES or DEBUG:
        settings = kwargs.pop('template_settings', {})
        if isinstance(tpl, adapter):
            TEMPLATES[tplid] = tpl
            if settings: TEMPLATES[tplid].prepare(**settings)
        elif "\n" in tpl or "{" in tpl or "%" in tpl or '$' in tpl:
            TEMPLATES[tplid] = adapter(source=tpl, lookup=lookup, **settings)
        else:
            TEMPLATES[tplid] = adapter(name=tpl, lookup=lookup, **settings)
    if not TEMPLATES[tplid]:
        abort(500, 'Template (%s) not found' % tpl)
    for dictarg in args[1:]: kwargs.update(dictarg)
    return TEMPLATES[tplid].render(kwargs)

mako_template = functools.partial(template, template_adapter=MakoTemplate)
cheetah_template = functools.partial(template, template_adapter=CheetahTemplate)
jinja2_template = functools.partial(template, template_adapter=Jinja2Template)


def view(tpl_name, **defaults):
    ''' Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, (dict, DictMixin)):
                tplvars = defaults.copy()
                tplvars.update(result)
                return template(tpl_name, **tplvars)
            elif result is None:
                return template(tpl_name, defaults)
            return result
        return wrapper
    return decorator

mako_view = functools.partial(view, template_adapter=MakoTemplate)
cheetah_view = functools.partial(view, template_adapter=CheetahTemplate)
jinja2_view = functools.partial(view, template_adapter=Jinja2Template)






###############################################################################
# Constants and Globals ########################################################
###############################################################################


TEMPLATE_PATH = ['./', './views/', '../views/']
TEMPLATES = {}
DEBUG = False
NORUN = False # If set, run() does nothing. Used by load_app()

#: A dict to map HTTP status codes (e.g. 404) to phrases (e.g. 'Not Found')
HTTP_CODES = httplib.responses
HTTP_CODES[418] = "I'm a teapot" # RFC 2324
HTTP_CODES[422] = "Unprocessable Entity" # RFC 4918
HTTP_CODES[428] = "Precondition Required"
HTTP_CODES[429] = "Too Many Requests"
HTTP_CODES[431] = "Request Header Fields Too Large"
HTTP_CODES[511] = "Network Authentication Required"
_HTTP_STATUS_LINES = dict((k, '%d %s'%(k,v)) for (k,v) in HTTP_CODES.items())

#: The default template used for error pages. Override with @error()
ERROR_PAGE_TEMPLATE = """
%%try:
    %%from %s import DEBUG, HTTP_CODES, request, touni
    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
    <html>
        <head>
            <title>Error: {{e.status}}</title>
            <style type="text/css">
              html {background-color: #eee; font-family: sans;}
              body {background-color: #fff; border: 1px solid #ddd;
                    padding: 15px; margin: 15px;}
              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}
            </style>
        </head>
        <body>
            <h1>Error: {{e.status}}</h1>
            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>
               caused an error:</p>
            <pre>{{e.body}}</pre>
            %%if DEBUG and e.exception:
              <h2>Exception:</h2>
              <pre>{{repr(e.exception)}}</pre>
            %%end
            %%if DEBUG and e.traceback:
              <h2>Traceback:</h2>
              <pre>{{e.traceback}}</pre>
            %%end
        </body>
    </html>
%%except ImportError:
    <b>ImportError:</b> Could not generate the error page. Please add bottle to
    the import path.
%%end
""" % __name__

#: A thread-safe instance of :class:`LocalRequest`. If accessed from within a
#: request callback, this instance always refers to the *current* request
#: (even on a multithreaded server).
request = LocalRequest()

#: A thread-safe instance of :class:`LocalResponse`. It is used to change the
#: HTTP response for the *current* request.
response = LocalResponse()

#: A thread-safe namespace. Not used by Bottle.
local = threading.local()

# Initialize app stack (create first empty Bottle app)
# BC: 0.6.4 and needed for run()
app = default_app = AppStack()
app.push()

#: A virtual package that redirects import statements.
#: Example: ``import bottle.ext.sqlite`` actually imports `bottle_sqlite`.
ext = _ImportRedirect('bottle.ext' if __name__ == '__main__' else __name__+".ext", 'bottle_%s').module

if __name__ == '__main__':
    opt, args, parser = _cmd_options, _cmd_args, _cmd_parser
    if opt.version:
        _stdout('Bottle %s\n'%__version__)
        sys.exit(0)
    if not args:
        parser.print_help()
        _stderr('\nError: No application specified.\n')
        sys.exit(1)

    sys.path.insert(0, '.')
    sys.modules.setdefault('bottle', sys.modules['__main__'])

    host, port = (opt.bind or 'localhost'), 8080
    if ':' in host and host.rfind(']') < host.rfind(':'):
        host, port = host.rsplit(':', 1)
    host = host.strip('[]')

    run(args[0], host=host, port=int(port), server=opt.server,
        reloader=opt.reload, plugins=opt.plugin, debug=opt.debug)




# THE END
Û
∑∆Xc           @Ä  sá  d  Z  d d l m Z d Z d Z d Z e d k r5d d l m Z e d d	 É Z	 e	 j
 Z e d
 d d d d Ée d d d d d d Ée d d d d d d Ée d d d d d d Ée d d d d d Ée d d d d d  Ée	 j É  \ Z Z e j oe j j d! É r2d d" l Z e j j É  n  n  d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l  Z  d d" l! Z! d d" l" Z" d d" l# Z# d d" l$ Z$ d d" l% Z% d d# l& m' Z( m& Z& m) Z) d d$ l" m* Z* d d% l+ m, Z, m- Z- d d& l. m/ Z/ d d' l0 m1 Z1 y d d( l2 m3 Z4 m5 Z6 Wn| e7 k
 rÔy d d( l8 m3 Z4 m5 Z6 WnN e7 k
 rÎy d d( l9 m3 Z4 m5 Z6 Wn  e7 k
 rÁd) Ñ  Z4 e4 Z6 n Xn Xn Xe! j: Z; e; d* d+ d+ f k Z< e; d, d- d+ f k  Z= d* d. d+ f e; k oLd* d, d+ f k  n Z> d/ Ñ  Z? y" e! j@ jA e! jB jA f \ ZC ZD Wn# eE k
 r°d0 Ñ  ZC d1 Ñ  ZD n Xe< rÜd d" lF jG ZH d d" lI ZJ d d2 lK mL ZL mM ZN d d3 lK mO ZO mP ZQ mR ZS e jT eS d4 d5 ÉZS d d6 lU mV ZV d d7 lW mX ZY d d" lZ ZZ d d8 l[ m\ Z\ d d9 l] m^ Z^ e_ Z` e_ Za d: Ñ  Zb d; Ñ  Zc ed Ze d< Ñ  Zf nd d" lH ZH d d" lJ ZJ d d2 lg mL ZL mM ZN d d3 lh mO ZO mP ZQ mR ZS d d6 li mV ZV d d= l me Ze d d" lj ZZ d d> lk mk Z\ d d? l^ ml Z^ e= rZd@ Zm e% jn em eo É d dA lp mY ZY dB Ñ  Zq e_ Zr n d d7 lW mX ZY ea Za e6 Zb es et dC dD dE É É dF dG Ñ Zu dF dH dI Ñ Zv e< r≥ev n eu Zw e> rËd dJ l[ mx Zx dK ex f dL Ñ  É  YZy n  dM Ñ  Zz e{ dN Ñ Z| dO Ñ  Z} dP e~ f dQ Ñ  É  YZ dR e~ f dS Ñ  É  YZÄ dT e~ f dU Ñ  É  YZÅ dV eÇ f dW Ñ  É  YZÉ dX eÉ f dY Ñ  É  YZÑ dZ eÉ f d[ Ñ  É  YZÖ d\ eÑ f d] Ñ  É  YZÜ d^ eÑ f d_ Ñ  É  YZá d` eÑ f da Ñ  É  YZà db Ñ  Zâ dc e~ f dd Ñ  É  YZä de e~ f df Ñ  É  YZã dg e~ f dh Ñ  É  YZå di e~ f dj Ñ  É  YZç dk Ñ  Zé dl e~ f dm Ñ  É  YZè dn e~ f do Ñ  É  YZê eë dp Ñ Zí dq eç f dr Ñ  É  YZì ds eê f dt Ñ  É  YZî eç Zï eê Zñ du eñ eÉ f dv Ñ  É  YZó dw eó f dx Ñ  É  YZò dy eÉ f dz Ñ  É  YZô d{ e~ f d| Ñ  É  YZö d} e~ f d~ Ñ  É  YZõ d e~ f dÄ Ñ  É  YZú dÅ eY f dÇ Ñ  É  YZù dÉ eù f dÑ Ñ  É  YZû dÖ eù f dÜ Ñ  É  YZü dá eY f dà Ñ  É  YZ† dâ e° f dä Ñ  É  YZ¢ dã e£ f då Ñ  É  YZ§ dç e~ f dé Ñ  É  YZ• dè e~ f dê Ñ  É  YZ¶ dë e~ f dí Ñ  É  YZß dì e~ f dî Ñ  É  YZ® dï dñ dó Ñ Z© eë dò Ñ Z™ dô dô dö Ñ Z´ dõ e{ dú dù Ñ Z¨ e≠ dû Ñ ZÆ dü Ñ  ZØ d† Ñ  Z∞ d° Ñ  Z± d+ d¢ Ñ Z≤ d£ Ñ  Z≥ d§ Ñ  Z¥ d• Ñ  Zµ d¶ Ñ  Z∂ dß Ñ  Z∑ d® Ñ  Z∏ d© Ñ  Zπ d™ Ñ  Z∫ d. d´ Ñ Zª d¨ d≠ dÆ Ñ Zº dØ Ñ  ZΩ eΩ d∞ É Zæ eΩ d± É Zø eΩ d≤ É Z¿ eΩ d≥ É Z¡ eΩ d¥ É Z¬ eΩ dµ É Z√ eΩ d∂ É Zƒ eΩ d∑ É Z≈ eΩ d∏ É Z∆ eΩ dπ É Z« eΩ d∫ É Z» dª e~ f dº Ñ  É  YZ… dΩ e… f dæ Ñ  É  YZ  dø e… f d¿ Ñ  É  YZÀ d¡ e… f d¬ Ñ  É  YZÃ d√ e… f dƒ Ñ  É  YZÕ d≈ e… f d∆ Ñ  É  YZŒ d« e… f d» Ñ  É  YZœ d… e… f d  Ñ  É  YZ– dÀ e… f dÃ Ñ  É  YZ— dÕ e… f dŒ Ñ  É  YZ“ dœ e… f d– Ñ  É  YZ” d— e… f d“ Ñ  É  YZ‘ d” e… f d‘ Ñ  É  YZ’ d’ e… f d÷ Ñ  É  YZ÷ d◊ e… f dÿ Ñ  É  YZ◊ dŸ e… f d⁄ Ñ  É  YZÿ d€ e… f d‹ Ñ  É  YZŸ d› e… f dﬁ Ñ  É  YZ⁄ dﬂ e… f d‡ Ñ  É  YZ€ d· e… f d‚ Ñ  É  YZ‹ i e  d„ 6eÀ d‰ 6eÃ d 6eŒ dÂ 6eÕ dÊ 6eœ dÁ 6e— dË 6e“ dÈ 6e” dÍ 6e‘ dÎ 6e’ dÏ 6e– dÌ 6eÿ dÓ 6eŸ dÔ 6e÷ d! 6e◊ d 6e⁄ dÒ 6e€ dÚ 6e‹ dõ 6Z› dÛ Ñ  Zﬁ dÙ Ñ  Zﬂ eÆ Z‡ eë d dı dˆ d. e{ e{ eë eë d˜ Ñ	 Z· d¯ e# j‚ f d˘ Ñ  É  YZ„ d˙ eò f d˚ Ñ  É  YZ‰ d¸ e~ f d˝ Ñ  É  YZÂ d˛ eÂ f dˇ Ñ  É  YZÊ d eÂ f dÑ  É  YZÁ deÂ f dÑ  É  YZË deÂ f dÑ  É  YZÈ de‰ f dÑ  É  YZÍ de~ f d	Ñ  É  YZÎ d
Ñ  ZÏ e jT eÏ deÊ ÉZÌ e jT eÏ deÁ ÉZÓ e jT eÏ deË ÉZÔ dÑ  Z e jT e deÊ ÉZÒ e jT e deÁ ÉZÚ e jT e deË ÉZÛ dddg ZÙ i  Zı e{ aˆ e{ a˜ eH j¯ Z˘ de˘ d<de˘ d<de˘ d<de˘ d<de˘ d<de˘ d<e° dÑ  e˘ j˙ É  DÉ É Z˚ de Z¸ eì É  Z˝ eî É  Z˛ e# jˇ É  Zˇ e§ É  Z Ze jÉ  eú e d k rdn e dd É jZe d k rÉe e e	 f \ ZZZejrueC d!e É e! j	d+ É n  er†ej
É  eD d"É e! j	d. É n  e! jjd+ d#É e! jjd$e! jd É ejpŸd%dˆ f \ ZZd&ek oejd'É ejd&É k  r-ejd&d. É \ ZZn  ejd(É Ze· ed+ d)ed*eeÉ d+ej d,ejd-ejd.ejÆ Én  d" S(/  sÕ  
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
iˇˇˇˇ(   t   with_statements   Marcel Hellkamps   0.12.9t   MITt   __main__(   t   OptionParsert   usages)   usage: %prog [options] package.module:apps	   --versiont   actiont
   store_truet   helps   show version number.s   -bs   --bindt   metavart   ADDRESSs   bind socket to ADDRESS.s   -ss   --servert   defaultt   wsgirefs   use SERVER as backend.s   -ps   --plugint   appends   install additional plugin/s.s   --debugs   start server in debug mode.s   --reloads   auto-reload on file changes.t   geventN(   t   datet   datetimet	   timedelta(   t   TemporaryFile(   t
   format_exct	   print_exc(   t
   getargspec(   t	   normalize(   t   dumpst   loadsc         CÄ  s   t  d É Ç d  S(   Ns/   JSON support requires Python 2.6 or simplejson.(   t   ImportError(   t   data(    (    s&   /home/lgardner/git/professor/bottle.pyt
   json_dumps6   s    i   i    i   i   i   c           CÄ  s   t  j É  d S(   Ni   (   t   syst   exc_info(    (    (    s&   /home/lgardner/git/professor/bottle.pyt   _eE   s    c         CÄ  s   t  j j |  É S(   N(   R   t   stdoutt   write(   t   x(    (    s&   /home/lgardner/git/professor/bottle.pyt   <lambda>L   s    c         CÄ  s   t  j j |  É S(   N(   R   t   stderrR   (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   M   s    (   t   urljoint   SplitResult(   t	   urlencodet   quotet   unquotet   encodingt   latin1(   t   SimpleCookie(   t   MutableMapping(   t   BytesIO(   t   ConfigParserc         CÄ  s   t  t |  É É S(   N(   t   json_ldst   touni(   t   s(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]   s    c         CÄ  s   t  |  d É S(   Nt   __call__(   t   hasattr(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ^   s    c          GÄ  s%   |  d |  d É j  |  d É Ç d  S(   Ni    i   i   (   t   with_traceback(   t   a(    (    s&   /home/lgardner/git/professor/bottle.pyt   _raise`   s    (   t   imap(   t   StringIO(   t   SafeConfigParsers?   Python 2.5 support may be dropped in future versions of Bottle.(   t	   DictMixinc         CÄ  s
   |  j  É  S(   N(   t   next(   t   it(    (    s&   /home/lgardner/git/professor/bottle.pyR:   o   s    s&   def _raise(*a): raise a[0], a[1], a[2]s   <py3fix>t   exect   utf8c         CÄ  s&   t  |  t É r |  j | É St |  É S(   N(   t
   isinstancet   unicodet   encodet   bytes(   R0   t   enc(    (    s&   /home/lgardner/git/professor/bottle.pyt   tobx   s    t   strictc         CÄ  s)   t  |  t É r |  j | | É St |  É S(   N(   R>   RA   t   decodeR?   (   R0   RB   t   err(    (    s&   /home/lgardner/git/professor/bottle.pyR/   z   s    (   t   TextIOWrappert   NCTextIOWrapperc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s   d  S(   N(    (   t   self(    (    s&   /home/lgardner/git/professor/bottle.pyt   closeÉ   s    (   t   __name__t
   __module__RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRH   Ç   s   c         OÄ  s2   y t  j |  | | | é Wn t k
 r- n Xd  S(   N(   t	   functoolst   update_wrappert   AttributeError(   t   wrappert   wrappedR4   t   ka(    (    s&   /home/lgardner/git/professor/bottle.pyRN   á   s      c         CÄ  s   t  j |  t d d Éd  S(   Nt
   stackleveli   (   t   warningst   warnt   DeprecationWarning(   t   messaget   hard(    (    s&   /home/lgardner/git/professor/bottle.pyt   deprê   s    c         CÄ  s:   t  |  t t t t f É r% t |  É S|  r2 |  g Sg  Sd  S(   N(   R>   t   tuplet   listt   sett   dict(   R   (    (    s&   /home/lgardner/git/professor/bottle.pyt   makelistì   s
     
 t   DictPropertyc           BÄ  sA   e  Z d  Z d e d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 RS(   s=    Property that maps to a key in a local dict-like attribute. c         CÄ  s!   | | | |  _  |  _ |  _ d  S(   N(   t   attrt   keyt	   read_only(   RI   R`   Ra   Rb   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __init__õ   s    c         CÄ  s9   t  j |  | d g  É| |  j p( | j |  _ |  _ |  S(   Nt   updated(   RM   RN   Ra   RK   t   getter(   RI   t   func(    (    s&   /home/lgardner/git/professor/bottle.pyR1   û   s    c         CÄ  sV   | d  k r |  S|  j t | |  j É } } | | k rN |  j | É | | <n  | | S(   N(   t   NoneRa   t   getattrR`   Re   (   RI   t   objt   clsRa   t   storage(    (    s&   /home/lgardner/git/professor/bottle.pyt   __get__£   s      c         CÄ  s5   |  j  r t d É Ç n  | t | |  j É |  j <d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   t   value(    (    s&   /home/lgardner/git/professor/bottle.pyt   __set__©   s    	 c         CÄ  s2   |  j  r t d É Ç n  t | |  j É |  j =d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   __delete__≠   s    	 N(
   RK   RL   t   __doc__Rg   t   FalseRc   R1   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR_   ô   s   			t   cached_propertyc           BÄ  s    e  Z d  Z d Ñ  Z d Ñ  Z RS(   s•    A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. c         CÄ  s   t  | d É |  _ | |  _ d  S(   NRp   (   Rh   Rp   Rf   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ∑   s    c         CÄ  s4   | d  k r |  S|  j | É } | j |  j j <| S(   N(   Rg   Rf   t   __dict__RK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   ª   s      (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRr   ≤   s   	t   lazy_attributec           BÄ  s    e  Z d  Z d Ñ  Z d Ñ  Z RS(   s4    A property that caches itself to the class object. c         CÄ  s#   t  j |  | d g  É| |  _ d  S(   NRd   (   RM   RN   Re   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   √   s    c         CÄ  s&   |  j  | É } t | |  j | É | S(   N(   Re   t   setattrRK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   «   s    (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt   ¡   s   	t   BottleExceptionc           BÄ  s   e  Z d  Z RS(   s-    A base class for exceptions used by bottle. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRv   ÷   s   t
   RouteErrorc           BÄ  s   e  Z d  Z RS(   s9    This is a base class for all routing related exceptions (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw   ‰   s   t
   RouteResetc           BÄ  s   e  Z d  Z RS(   sf    If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx   Ë   s   t   RouterUnknownModeErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRy   Ï   s    t   RouteSyntaxErrorc           BÄ  s   e  Z d  Z RS(   s@    The route parser found something not supported by this router. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRz   Ô   s   t   RouteBuildErrorc           BÄ  s   e  Z d  Z RS(   s    The route could not be built. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{   Û   s   c         CÄ  s&   d |  k r |  St  j d d Ñ  |  É S(   s^    Turn all capturing groups in a regular expression pattern into
        non-capturing groups. t   (s   (\\*)(\(\?P<[^>]+>|\((?!\?))c         SÄ  s7   t  |  j d É É d r& |  j d É S|  j d É d S(   Ni   i   i    s   (?:(   t   lent   group(   t   m(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ¸   s    (   t   ret   sub(   t   p(    (    s&   /home/lgardner/git/professor/bottle.pyt   _re_flatten˜   s     	t   Routerc           BÄ  st   e  Z d  Z d Z d Z d Z e d Ñ Z d Ñ  Z e	 j
 d É Z d Ñ  Z d d Ñ Z d	 Ñ  Z d
 Ñ  Z d Ñ  Z RS(   sA   A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    s   [^/]+RÄ   ic   c         Ä  sz   g  à  _  i  à  _ i  à  _ i  à  _ i  à  _ i  à  _ | à  _ i á  f d Ü  d 6d Ñ  d 6d Ñ  d 6d Ñ  d 6à  _ d  S(	   Nc         Ä  s   t  |  p à  j É d  d  f S(   N(   RÉ   t   default_patternRg   (   t   conf(   RI   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    RÄ   c         SÄ  s   d t  d Ñ  f S(   Ns   -?\d+c         SÄ  s   t  t |  É É S(   N(   t   strt   int(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   Rà   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    Rà   c         SÄ  s   d t  d Ñ  f S(   Ns   -?[\d.]+c         SÄ  s   t  t |  É É S(   N(   Rá   t   float(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   Râ   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    Râ   c         SÄ  s   d S(   Ns   .+?(   s   .+?NN(   Rg   (   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR!      s    t   path(   t   rulest   _groupst   buildert   statict   dyna_routest   dyna_regexest   strict_ordert   filters(   RI   RD   (    (   RI   s&   /home/lgardner/git/professor/bottle.pyRc     s    							

c         CÄ  s   | |  j  | <d S(   s‚    Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. N(   Rí   (   RI   t   nameRf   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   add_filter"  s    sÄ   (\\*)(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)(?::((?:\\.|[^\\>]+)+)?)?)?>))c   	      cÄ  s?  d \ } } x˚ |  j  j | É D]Á } | | | | j É  !7} | j É  } t | d É d rè | | j d É t | d É 7} | j É  } q n  | r¶ | d  d  f Vn  | d d  k r√ | d d !n
 | d d !\ } } } | | pÂ d | pÓ d  f V| j É  d } } q W| t | É k s"| r;| | | d  d  f Vn  d  S(	   Ni    t    i   i   i   i   R
   (   i    Rï   (   t   rule_syntaxt   finditert   startt   groupsR}   R~   t   endRg   (	   RI   t   rulet   offsett   prefixt   matcht   gRì   t   filtrRÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _itertokens-  s    !3c         Ä  s˛  d } g  } d } g  â  g  } t  }	 x|  j | É D]\ }
 } } | rt }	 | d k rg |  j } n  |  j | | É \ } } } |
 sÆ | d | 7} d | }
 | d 7} n! | d |
 | f 7} | j |
 É | rÎ à  j |
 | f É n  | j |
 | p˝ t f É q4 |
 r4 | t j |
 É 7} | j d |
 f É q4 q4 W| |  j
 | <| r]| |  j
 | <n  |	 r§|  j r§|  j j | i  É | d f |  j | |  j | É <d Sy  t j d	 | É } | j â Wn- t j k
 rÛt d
 | t É  f É Ç n Xà  rá  á f d Ü  } n! | j r*á f d Ü  } n d } t | É } | | | | f } | | f |  j k r≠t råd } t j | | | f t É n  | |  j | |  j | | f <n@ |  j j | g  É j | É t |  j | É d |  j | | f <|  j | É d S(   s<    Add a new rule or replace the target for an existing rule. i    Rï   R
   s   (?:%s)s   anon%di   s
   (?P<%s>%s)Ns   ^(%s)$s   Could not add Route: %s (%s)c         Ä  sh   à |  É j  É  } xO à  D]G \ } } y | | | É | | <Wq t k
 r_ t d d É Ç q Xq W| S(   Niê  s   Path has wrong format.(   t	   groupdictt
   ValueErrort	   HTTPError(   Rä   t   url_argsRì   t   wildcard_filter(   Rí   t   re_match(    s&   /home/lgardner/git/professor/bottle.pyt   getargsh  s    c         Ä  s   à  |  É j  É  S(   N(   R¢   (   Rä   (   Rß   (    s&   /home/lgardner/git/professor/bottle.pyR®   q  s    s3   Route <%s %s> overwrites a previously defined route(   t   TrueR°   Rq   t   default_filterRí   R   Rá   RÄ   t   escapeRg   Rç   Rë   Ré   t
   setdefaultt   buildt   compileRû   t   errorRz   R   t
   groupindexRÉ   Rå   t   DEBUGRT   RU   t   RuntimeWarningRè   R}   t   _compile(   RI   Rõ   t   methodt   targetRì   t   anonst   keyst   patternRç   t	   is_staticRa   t   modeRÜ   t   maskt	   in_filtert
   out_filtert
   re_patternR®   t   flatpatt
   whole_rulet   msg(    (   Rí   Rß   s&   /home/lgardner/git/professor/bottle.pyt   add>  sf     
   	!$c         CÄ  sÿ   |  j  | } g  } |  j | <|  j } x™ t d t | É | É D]ê } | | | | !} d Ñ  | DÉ } d j d Ñ  | DÉ É } t j | É j } g  | D] \ } } }	 }
 |	 |
 f ^ qô } | j	 | | f É q@ Wd  S(   Ni    c         sÄ  s!   |  ] \ } } } } | Vq d  S(   N(    (   t   .0t   _Rø   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>ä  s    t   |c         sÄ  s   |  ] } d  | Vq d S(   s   (^%s$)N(    (   R√   Rø   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>ã  s    (
   Rè   Rê   t   _MAX_GROUPS_PER_PATTERNt   rangeR}   t   joinRÄ   RÆ   Rû   R   (   RI   R¥   t	   all_rulest
   comborulest	   maxgroupsR    t   somet   combinedRƒ   Rµ   R®   Rã   (    (    s&   /home/lgardner/git/professor/bottle.pyR≥   Ñ  s    	+c   
      OÄ  sÍ   |  j  j | É } | s* t d | É Ç n  yé x( t | É D] \ } } | | d | <q: Wd j g  | D]- \ } } | rå | | j | É É n | ^ qe É }	 | s• |	 S|	 d t | É SWn+ t k
 rÂ t d t É  j	 d É Ç n Xd S(   s2    Build an URL by filling the wildcards in a rule. s   No route with that name.s   anon%dRï   t   ?s   Missing URL argument: %ri    N(
   Rç   t   getR{   t	   enumerateR»   t   popR%   t   KeyErrorR   t   args(
   RI   t   _nameR∂   t   queryRç   t   iRm   t   nt   ft   url(    (    s&   /home/lgardner/git/professor/bottle.pyR≠   ê  s      C c         CÄ  s<  | d j  É  } | d p d } d } | d k rG d | d d g } n d | d g } xÿ | D]– } | |  j k r∏ | |  j | k r∏ |  j | | \ } } | | r± | | É n i  f S| |  j k r] xc |  j | D]Q \ } }	 | | É }
 |
 r’ |	 |
 j d \ } } | | r| | É n i  f Sq’ Wq] q] Wt g  É } t | É } x> t |  j É | D]) } | |  j | k r]| j | É q]q]Wx_ t |  j É | | D]F } x= |  j | D]. \ } }	 | | É }
 |
 r∂| j | É q∂q∂Wq¢W| rd	 j t | É É } t	 d
 d d | ÉÇ n  t	 d d t
 | É É Ç d S(   sD    Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). t   REQUEST_METHODt	   PATH_INFOt   /t   HEADt   PROXYt   GETt   ANYi   t   ,iï  s   Method not allowed.t   Allowiî  s   Not found: N(   t   upperRg   Ré   Rê   t	   lastindexR\   R¬   R»   t   sortedR§   t   repr(   RI   t   environt   verbRä   Rµ   t   methodsR¥   R®   RÕ   Rã   Rû   t   allowedt   nocheckt   allow_header(    (    s&   /home/lgardner/git/professor/bottle.pyRû   õ  s<    "'N(   RK   RL   Rp   RÖ   R™   R∆   Rq   Rc   Rî   RÄ   RÆ   Rñ   R°   Rg   R¬   R≥   R≠   Rû   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÑ   ˇ   s   
		F		t   Routec           BÄ  sí   e  Z d  Z d d d d Ñ Z d Ñ  Z e d Ñ  É Z d Ñ  Z d Ñ  Z	 e
 d Ñ  É Z d Ñ  Z d Ñ  Z d	 Ñ  Z d
 Ñ  Z d d Ñ Z d Ñ  Z RS(   sÓ    This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    c   	      KÄ  sp   | |  _  | |  _ | |  _ | |  _ | p- d  |  _ | p< g  |  _ | pK g  |  _ t É  j	 | d t
 É|  _ d  S(   Nt   make_namespaces(   t   appRõ   R¥   t   callbackRg   Rì   t   pluginst   skiplistt
   ConfigDictt	   load_dictR©   t   config(	   RI   RÔ   Rõ   R¥   R   Rì   RÒ   RÚ   Rı   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Õ  s    				c         OÄ  s   t  d É |  j | | é  S(   Nsî   Some APIs changed to return Route() instances instead of callables. Make sure to use the Route.call method and not to call Route instances directly.(   RY   t   call(   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ‚  s    
c         CÄ  s
   |  j  É  S(   sç    The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.(   t   _make_callback(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRˆ   Ë  s    c         CÄ  s   |  j  j d d É d S(   sk    Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. Rˆ   N(   Rs   R—   Rg   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   resetÓ  s    c         CÄ  s   |  j  d S(   s:    Do all on-demand work immediately (useful for debugging).N(   Rˆ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   prepareÛ  s    c         CÄ  sY   t  d É t d |  j d |  j d |  j d |  j d |  j d |  j d |  j d	 |  j	 É S(
   Ns=   Switch to Plugin API v2 and access the Route object directly.Rõ   R¥   R   Rì   RÔ   Rı   t   applyt   skip(
   RY   R]   Rõ   R¥   R   Rì   RÔ   Rı   RÒ   RÚ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _context˜  s    
!c         cÄ  s¬   t  É  } x≤ t |  j j |  j É D]ó } t |  j k r< Pn  t | d t É } | ru | |  j k s# | | k ru q# n  | |  j k s# t | É |  j k rü q# n  | rµ | j	 | É n  | Vq# Wd S(   s)    Yield all Plugins affecting this route. Rì   N(
   R\   t   reversedRÔ   RÒ   R©   RÚ   Rh   Rq   t   typeR¬   (   RI   t   uniqueRÇ   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   all_plugins˛  s    	  ! $  c         CÄ  s¬   |  j  } x≤ |  j É  D]§ } ya t | d É rp t | d d É } | d k rR |  n |  j } | j | | É } n | | É } Wn t k
 ró |  j É  SX| |  j  k	 r t | |  j  É q q W| S(   NR˙   t   apii   (	   R   R   R2   Rh   R¸   R˙   Rx   R˜   RN   (   RI   R   t   pluginR  t   context(    (    s&   /home/lgardner/git/professor/bottle.pyR˜   	  s    	c         CÄ  sx   |  j  } t | t r d n d | É } t r3 d n d } x8 t | | É rs t | | É rs t | | É d j } q< W| S(   sq    Return the callback. If the callback is a decorated function, try to
            recover the original function. t   __func__t   im_funct   __closure__t   func_closurei    (   R   Rh   t   py3kR2   t   cell_contents(   RI   Rf   t   closure_attr(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_undecorated_callback  s    	!c         CÄ  s   t  |  j É  É d S(   s”    Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. i    (   R   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   get_callback_args#  s    c         CÄ  s8   x1 |  j  |  j j f D] } | | k r | | Sq W| S(   sp    Lookup a config field and return its value, first checking the
            route.config, then route.app.config.(   Rı   RÔ   t   conifg(   RI   Ra   R
   RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_config)  s     c         CÄ  s#   |  j  É  } d |  j |  j | f S(   Ns
   <%s %r %r>(   R  R¥   Rõ   (   RI   t   cb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __repr__0  s    N(   RK   RL   Rp   Rg   Rc   R1   Rr   Rˆ   R¯   R˘   t   propertyR¸   R   R˜   R  R  R  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÌ   «  s   						
	t   Bottlec           BÄ  s[  e  Z d  Z e e d Ñ Z e d d É Z d& Z d Z e	 d Ñ  É Z
 d Ñ  Z d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d' d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d' d d' d' d' d' d Ñ Z d' d d Ñ Z d' d d Ñ Z d' d d Ñ Z d' d d Ñ Z d d  Ñ Z d! Ñ  Z  d" Ñ  Z! d' d# Ñ Z" d$ Ñ  Z# d% Ñ  Z$ RS((   s^   Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    c         CÄ  s‘   t  É  |  _ t j |  j d É |  j _ |  j j d d t É |  j j d d t É | |  j d <| |  j d <t É  |  _	 g  |  _
 t É  |  _ i  |  _ g  |  _ |  j d r¿ |  j t É  É n  |  j t É  É d  S(   NRı   t   autojsont   validatet   catchall(   RÛ   Rı   RM   t   partialt   trigger_hookt
   _on_changet   meta_sett   boolt   ResourceManagert	   resourcest   routesRÑ   t   routert   error_handlerRÒ   t   installt
   JSONPlugint   TemplatePlugin(   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   G  s    			Rı   R  t   before_requestt   after_requestt	   app_resetc         CÄ  s   t  d Ñ  |  j DÉ É S(   Nc         sÄ  s   |  ] } | g  f Vq d  S(   N(    (   R√   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>f  s    (   R]   t   _Bottle__hook_names(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hooksd  s    c         CÄ  sA   | |  j  k r) |  j | j d | É n |  j | j | É d S(   s´   Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        i    N(   t   _Bottle__hook_reversedR'  t   insertR   (   RI   Rì   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   add_hookh  s    c         CÄ  s>   | |  j  k r: | |  j  | k r: |  j  | j | É t Sd S(   s     Remove a callback from a hook. N(   R'  t   removeR©   (   RI   Rì   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   remove_hookx  s    "c         OÄ  s(   g  |  j  | D] } | | | é  ^ q S(   s.    Trigger a hook and return a list of results. (   R'  (   RI   t   _Bottle__nameR”   t   kwargst   hook(    (    s&   /home/lgardner/git/professor/bottle.pyR  ~  s    c         Ä  s   á  á f d Ü  } | S(   se    Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details.c         Ä  s   à j  à  |  É |  S(   N(   R*  (   Rf   (   Rì   RI   (    s&   /home/lgardner/git/professor/bottle.pyt	   decoratorÖ  s    (    (   RI   Rì   R0  (    (   Rì   RI   s&   /home/lgardner/git/professor/bottle.pyR/  Ç  s    c         Ä  s  t  à  t É r t d t É n  g  | j d É D] } | r/ | ^ q/ } | s\ t d É Ç n  t | É â á  á f d Ü  } | j d t É | j d d É | j d i | d	 6à  d
 6É | | d <|  j d d j	 | É | ç | j
 d É s|  j d d j	 | É | ç n  d S(   sø   Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        s*   Parameter order of Bottle.mount() changed.R‹   s   Empty path prefix.c          Ä  sî   z~ t  j à É t g  É â  d  á  f d Ü }  à t  j |  É } | rg à  j rg t j à  j | É } n  | ps à  j à  _ à  SWd  t  j à É Xd  S(   Nc         Ä  s[   | r! z t  | å  Wd  d  } Xn  |  à  _ x$ | D] \ } } à  j | | É q1 Wà  j j S(   N(   R5   Rg   t   statust
   add_headert   bodyR   (   R1  t
   headerlistR   Rì   Rm   (   t   rs(    s&   /home/lgardner/git/professor/bottle.pyt   start_response°  s    
	 (   t   requestt
   path_shiftt   HTTPResponseRg   RÁ   R3  t	   itertoolst   chain(   R6  R3  (   RÔ   t
   path_depth(   R5  s&   /home/lgardner/git/professor/bottle.pyt   mountpoint_wrapperù  s    	 R˚   R¥   Rﬁ   t
   mountpointRù   Rµ   R   s   /%s/<:re:.*>N(   R>   t
   basestringRY   R©   t   splitR£   R}   R¨   t   routeR»   t   endswith(   RI   Rù   RÔ   t   optionsRÇ   t   segmentsR=  (    (   RÔ   R<  s&   /home/lgardner/git/professor/bottle.pyt   mountä  s    ( 
c         CÄ  s=   t  | t É r | j } n  x | D] } |  j | É q" Wd S(   sÙ    Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. N(   R>   R  R  t	   add_route(   RI   R  RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   merge∫  s    c         CÄ  si   t  | d É r | j |  É n  t | É rK t  | d É rK t d É Ç n  |  j j | É |  j É  | S(   s‚    Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        t   setupR˙   s.   Plugins must be callable or implement .apply()(   R2   RH  t   callablet	   TypeErrorRÒ   R   R¯   (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR   ƒ  s     
c         CÄ  sœ   g  | } } x® t  t |  j É É d d d Ö D]Ñ \ } } | t k s~ | | k s~ | t | É k s~ t | d t É | k r0 | j | É |  j | =t | d É r¥ | j É  q¥ q0 q0 W| rÀ |  j	 É  n  | S(   s)   Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. NiˇˇˇˇRì   RJ   (
   R[   R–   RÒ   R©   R˛   Rh   R   R2   RJ   R¯   (   RI   R  t   removedR+  R÷   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   uninstall–  s    /*
  c         CÄ  sì   | d k r |  j } n+ t | t É r3 | g } n |  j | g } x | D] } | j É  qJ Wt rÇ x | D] } | j É  qk Wn  |  j d É d S(   s™    Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. R%  N(   Rg   R  R>   RÌ   R¯   R±   R˘   R  (   RI   RA  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR¯   ﬂ  s        c         CÄ  s=   x- |  j  D]" } t | d É r
 | j É  q
 q
 Wt |  _ d S(   s2    Close the application and all installed plugins. RJ   N(   RÒ   R2   RJ   R©   t   stopped(   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   Î  s     c         KÄ  s   t  |  | ç d S(   s-    Calls :func:`run` with the same parameters. N(   t   run(   RI   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ò  s    c         CÄ  s   |  j  j | É S(   s›    Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match.(   R  Rû   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRû   ı  s    c         KÄ  sV   t  j j d d É j d É d } |  j j | | ç j d É } t t d | É | É S(   s,    Return a string that matches a named route t   SCRIPT_NAMERï   R‹   (   R7  RÁ   Rœ   t   stripR  R≠   t   lstripR#   (   RI   t	   routenamet   kargst
   scriptnamet   location(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_url˚  s    "c         CÄ  sL   |  j  j | É |  j j | j | j | d | j Ét rH | j É  n  d S(   sS    Add a route object, but do not change the :data:`Route.app`
            attribute.Rì   N(	   R  R   R  R¬   Rõ   R¥   Rì   R±   R˘   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyRF    s    % Rﬂ   c   	      Ä  si   t  à É r d à â } n  t | É â t | É â á  á á á á á á f d Ü  } | re | | É S| S(   s   A decorator to bind a function to a request URL. Example::

                @app.route('/hello/:name')
                def hello(name):
                    return 'Hello %s' % name

            The ``:name`` part is a wildcard. See :class:`Router` for syntax
            details.

            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.

            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        c         Ä  sü   t  |  t É r t |  É }  n  xz t à É p6 t |  É D]` } xW t à É D]I } | j É  } t à | | |  d à d à d à à  ç} à j | É qJ Wq7 W|  S(   NRì   RÒ   RÚ   (   R>   R?  t   loadR^   t   yieldroutesR„   RÌ   RF  (   R   Rõ   RË   RA  (   Rı   R¥   Rì   Rä   RÒ   RI   RÚ   (    s&   /home/lgardner/git/professor/bottle.pyR0  &  s     N(   RI  Rg   R^   (	   RI   Rä   R¥   R   Rì   R˙   R˚   Rı   R0  (    (   Rı   R¥   Rì   Rä   RÒ   RI   RÚ   s&   /home/lgardner/git/professor/bottle.pyRA    s     !
c         KÄ  s   |  j  | | | ç S(   s    Equals :meth:`route`. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   2  s    t   POSTc         KÄ  s   |  j  | | | ç S(   s8    Equals :meth:`route` with a ``POST`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   post6  s    t   PUTc         KÄ  s   |  j  | | | ç S(   s7    Equals :meth:`route` with a ``PUT`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   put:  s    t   DELETEc         KÄ  s   |  j  | | | ç S(   s:    Equals :meth:`route` with a ``DELETE`` method parameter. (   RA  (   RI   Rä   R¥   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete>  s    iÙ  c         Ä  s   á  á f d Ü  } | S(   s<    Decorator: Register an output handler for a HTTP error codec         Ä  s   |  à j  t à  É <|  S(   N(   R  Rà   (   t   handler(   t   codeRI   (    s&   /home/lgardner/git/professor/bottle.pyRP   D  s    (    (   RI   R`  RP   (    (   R`  RI   s&   /home/lgardner/git/professor/bottle.pyRØ   B  s    c         CÄ  s   t  t t d | ÉÉ S(   Nt   e(   RC   t   templatet   ERROR_PAGE_TEMPLATE(   RI   t   res(    (    s&   /home/lgardner/git/professor/bottle.pyt   default_error_handlerI  s    c         CÄ  sã  | d } | d <t  rY y  | j d É j d É | d <WqY t k
 rU t d d É SXn  yä |  | d <t j | É t j É  zT |  j d É |  j	 j
 | É \ } } | | d	 <| | d
 <| | d <| j | ç  SWd  |  j d É XWn° t k
 r˙ t É  St k
 r| j É  |  j | É St t t f k
 r:Ç  nM t k
 rÜ|  j sVÇ  n  t É  } | d j | É t d d t É  | É SXd  S(   NR€   s   bottle.raw_pathR)   R=   iê  s#   Invalid path string. Expected UTF-8s
   bottle.appR#  s   route.handles   bottle.routes   route.url_argsR$  s   wsgi.errorsiÙ  s   Internal Server Error(   R  R@   RE   t   UnicodeErrorR§   R7  t   bindt   responseR  R  Rû   Rˆ   R9  R   Rx   R¯   t   _handlet   KeyboardInterruptt
   SystemExitt   MemoryErrort	   ExceptionR  R   R   (   RI   RÁ   Rä   RA  R”   t
   stacktrace(    (    s&   /home/lgardner/git/professor/bottle.pyRi  L  s>     





	 	c         CÄ  s$  | s# d t  k r d t  d <n  g  St | t t f É rn t | d t t f É rn | d d d !j | É } n  t | t É rí | j t  j É } n  t | t É r« d t  k r¿ t	 | É t  d <n  | g St | t
 É r| j t  É |  j j | j |  j É | É } |  j | É St | t É r=| j t  É |  j | j É St | d É ròd t j k rlt j d | É St | d É sãt | d É ròt | É Sn  y5 t | É } t | É } x | sÀt | É } q∂WWnä t k
 rÍ|  j d É St k
 rt É  } nW t t t f k
 rÇ  n; t k
 rY|  j s;Ç  n  t
 d d	 t É  t  É  É } n Xt | t É rv|  j | É St | t É rùt! j" | g | É } n_ t | t É r÷d
 Ñ  } t# | t! j" | g | É É } n& d t$ | É } |  j t
 d | É É St | d É r t% | | j& É } n  | S(   s˛    Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        s   Content-Lengthi    t   reads   wsgi.file_wrapperRJ   t   __iter__Rï   iÙ  s   Unhandled exceptionc         SÄ  s   |  j  t j É S(   N(   R@   Rh  t   charset(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   Æ  s    s   Unsupported response type: %s('   Rh  R>   RZ   R[   RA   R?   R»   R@   Rq  R}   R§   R˙   R  Rœ   t   status_codeRe  t   _castR9  R3  R2   R7  RÁ   t   WSGIFileWrappert   iterR:   t   StopIterationR   Rj  Rk  Rl  Rm  R  R   R:  R;  R6   R˛   t
   _closeiterRJ   (   RI   t   outt   peekt   ioutt   firstt   new_itert   encoderR¡   (    (    s&   /home/lgardner/git/professor/bottle.pyRs  o  sh    !		 	!c         CÄ  sE  yw |  j  |  j | É É } t j d k s: | d d k r_ t | d É rV | j É  n  g  } n  | t j t j É | SWn« t t	 t
 f k
 rñ Ç  n´ t k
 r@|  j s≤ Ç  n  d t | j d	 d
 É É } t r| d t t t É  É É t t É  É f 7} n  | d j | É d g } | d | t j É  É t | É g SXd S(   s    The bottle WSGI-interface. id   ie   iÃ   i0  R⁄   R›   RJ   s4   <h1>Critical error while processing request: %s</h1>R€   R‹   sD   <h2>Error:</h2>
<pre>
%s
</pre>
<h2>Traceback:</h2>
<pre>
%s
</pre>
s   wsgi.errorss   Content-Types   text/html; charset=UTF-8s   500 INTERNAL SERVER ERRORN(   id   ie   iÃ   i0  (   s   Content-Types   text/html; charset=UTF-8(   Rs  Ri  Rh  t   _status_codeR2   RJ   t   _status_lineR4  Rj  Rk  Rl  Rm  R  t   html_escapeRœ   R±   RÊ   R   R   R   R   R   RC   (   RI   RÁ   R6  Rx  RF   t   headers(    (    s&   /home/lgardner/git/professor/bottle.pyt   wsgi∑  s.     		 )	c         CÄ  s   |  j  | | É S(   s9    Each instance of :class:'Bottle' is a WSGI application. (   RÇ  (   RI   RÁ   R6  (    (    s&   /home/lgardner/git/professor/bottle.pyR1   —  s    (   s   before_requests   after_requests	   app_resets   configN(%   RK   RL   Rp   R©   Rc   R_   R  R&  R(  Rr   R'  R*  R,  R  R/  RE  RG  R   RL  Rg   R¯   RJ   RN  Rû   RV  RF  RA  Rœ   RZ  R\  R^  RØ   Re  Ri  Rs  RÇ  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  >  s@   					0	
							)		#H	t   BaseRequestc           BÄ  s;  e  Z d  Z d Z d Z d@ d Ñ Z e d d d e Éd Ñ  É Z	 e d d d e Éd Ñ  É Z
 e d d	 d e Éd
 Ñ  É Z e d Ñ  É Z e d Ñ  É Z e d d d e Éd Ñ  É Z d@ d Ñ Z e d d d e Éd Ñ  É Z d@ d@ d Ñ Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z e d d d e Éd Ñ  É Z d Ñ  Z d Ñ  Z e d d d e Éd  Ñ  É Z d! Ñ  Z e d" Ñ  É Z e d# Ñ  É Z e Z e d d$ d e Éd% Ñ  É Z e d& Ñ  É Z  e d d' d e Éd( Ñ  É Z! e d) Ñ  É Z" e d* Ñ  É Z# e d+ Ñ  É Z$ d, d- Ñ Z% e d. Ñ  É Z& e d/ Ñ  É Z' e d0 Ñ  É Z( e d1 Ñ  É Z) e d2 Ñ  É Z* e d3 Ñ  É Z+ e d4 Ñ  É Z, d5 Ñ  Z- d@ d6 Ñ Z. d7 Ñ  Z/ d8 Ñ  Z0 d9 Ñ  Z1 d: Ñ  Z2 d; Ñ  Z3 d< Ñ  Z4 d= Ñ  Z5 d> Ñ  Z6 d? Ñ  Z7 RS(A   sd   A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    RÁ   i ê c         CÄ  s,   | d k r i  n | |  _ |  |  j d <d S(   s!    Wrap a WSGI environ dictionary. s   bottle.requestN(   Rg   RÁ   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Ï  s    s
   bottle.appRb   c         CÄ  s   t  d É Ç d S(   s+    Bottle application handling this request. s0   This request is not connected to an application.N(   t   RuntimeError(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÔ   Û  s    s   bottle.routec         CÄ  s   t  d É Ç d S(   s=    The bottle :class:`Route` object that matches this request. s)   This request is not connected to a route.N(   RÑ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRA  ¯  s    s   route.url_argsc         CÄ  s   t  d É Ç d S(   s'    The arguments extracted from the URL. s)   This request is not connected to a route.N(   RÑ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR•   ˝  s    c         CÄ  s    d |  j  j d d É j d É S(   sÜ    The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). R‹   R€   Rï   (   RÁ   Rœ   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRä     s    c         CÄ  s   |  j  j d d É j É  S(   s6    The ``REQUEST_METHOD`` value as an uppercase string. R⁄   Rﬂ   (   RÁ   Rœ   R„   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¥     s    s   bottle.request.headersc         CÄ  s   t  |  j É S(   sf    A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. (   t   WSGIHeaderDictRÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÅ    s    c         CÄ  s   |  j  j | | É S(   sA    Return the value of a request header, or a given default value. (   RÅ  Rœ   (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_header  s    s   bottle.request.cookiesc         CÄ  s5   t  |  j j d d É É j É  } t d Ñ  | DÉ É S(   så    Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. t   HTTP_COOKIERï   c         sÄ  s!   |  ] } | j  | j f Vq d  S(   N(   Ra   Rm   (   R√   t   c(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R*   RÁ   Rœ   t   valuest	   FormsDict(   RI   t   cookies(    (    s&   /home/lgardner/git/professor/bottle.pyRã    s    !c         CÄ  sY   |  j  j | É } | rO | rO t | | É } | rK | d | k rK | d S| S| pX | S(   s   Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. i    i   (   Rã  Rœ   t   cookie_decode(   RI   Ra   R
   t   secretRm   t   dec(    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_cookie  s
    "s   bottle.request.queryc         CÄ  sT   t  É  } |  j d <t |  j j d d É É } x | D] \ } } | | | <q6 W| S(   s    The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. s
   bottle.gett   QUERY_STRINGRï   (   Rä  RÁ   t
   _parse_qslRœ   (   RI   Rœ   t   pairsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR’   )  s
    s   bottle.request.formsc         CÄ  sI   t  É  } x9 |  j j É  D]( \ } } t | t É s | | | <q q W| S(   s   Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. (   Rä  RY  t   allitemsR>   t
   FileUpload(   RI   t   formsRì   t   item(    (    s&   /home/lgardner/git/professor/bottle.pyRï  5  s
    	s   bottle.request.paramsc         CÄ  sa   t  É  } x' |  j j É  D] \ } } | | | <q Wx' |  j j É  D] \ } } | | | <qC W| S(   sâ    A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. (   Rä  R’   Rì  Rï  (   RI   t   paramsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRó  A  s    	s   bottle.request.filesc         CÄ  sI   t  É  } x9 |  j j É  D]( \ } } t | t É r | | | <q q W| S(   sò    File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        (   Rä  RY  Rì  R>   Rî  (   RI   t   filesRì   Rñ  (    (    s&   /home/lgardner/git/professor/bottle.pyRò  L  s
    	s   bottle.request.jsonc         CÄ  sX   |  j  j d d É j É  j d É d } | d k rT |  j É  } | sJ d St | É Sd S(   sÚ    If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. t   CONTENT_TYPERï   t   ;i    s   application/jsonN(   RÁ   Rœ   t   lowerR@  t   _get_body_stringRg   t
   json_loads(   RI   t   ctypet   b(    (    s&   /home/lgardner/git/professor/bottle.pyt   jsonX  s    (
c         cÄ  sW   t  d |  j É } x> | rR | t | | É É } | s: Pn  | V| t | É 8} q Wd  S(   Ni    (   t   maxt   content_lengtht   minR}   (   RI   Ro  t   bufsizet   maxreadt   part(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _iter_bodyf  s    	 c         cÄ  sï  t  d d É } t d É t d É t d É } } } xYt rê| d É } xT | d | k r† | d É } | | 7} | sÇ | Ç n  t | É | k rM | Ç qM qM W| j | É \ }	 }
 }
 y t t |	 j É  É d É } Wn t k
 rÒ | Ç n X| d	 k rPn  | } xg | d	 k rq| s5| t	 | | É É } n  | |  | | } } | sY| Ç n  | V| t | É 8} qW| d
 É | k r8 | Ç q8 q8 Wd  S(   Niê  s*   Error while parsing chunked transfer body.s   
Rö  Rï   i   i˛ˇˇˇi   i    i   (
   R§   RC   R©   R}   t	   partitionRà   t   tonatRP  R£   R£  (   RI   Ro  R§  RF   t   rnt   semt   bst   headerRà  t   sizeRƒ   R•  t   buffR¶  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _iter_chunkedn  s:    &	
 	 
  	s   bottle.request.bodyc         CÄ  sÂ   |  j  r |  j n |  j } |  j d j } t É  d t } } } xÇ | | |  j É D]n } | j | É | t	 | É 7} | rU | |  j k rU t
 d d É | } } | j | j É  É ~ t } qU qU W| |  j d <| j d É | S(   Ns
   wsgi.inputi    R∫   s   w+b(   t   chunkedR∞  Rß  RÁ   Ro  R,   Rq   t   MEMFILE_MAXR   R}   R   t   getvalueR©   t   seek(   RI   t	   body_itert	   read_funcR3  t	   body_sizet   is_temp_fileR¶  t   tmp(    (    s&   /home/lgardner/git/professor/bottle.pyt   _bodyâ  s    c         CÄ  sÉ   |  j  } | |  j k r* t d d É Ç n  | d k  rF |  j d } n  |  j j | É } t | É |  j k r t d d É Ç n  | S(   s~    read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. iù  s   Request to largei    i   (   R¢  R≤  R§   R3  Ro  R}   (   RI   t   clenR   (    (    s&   /home/lgardner/git/professor/bottle.pyRú  ö  s    	 c         CÄ  s   |  j  j d É |  j  S(   sl   The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. i    (   R∫  R¥  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR3  ¶  s    c         CÄ  s   d |  j  j d d É j É  k S(   s(    True if Chunked transfer encoding was. R±  t   HTTP_TRANSFER_ENCODINGRï   (   RÁ   Rœ   Rõ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR±  ∞  s    s   bottle.request.postc   	      CÄ  sw  t  É  } |  j j d É s[ t t |  j É  d É É } x | D] \ } } | | | <q= W| Si d d 6} x1 d D]) } | |  j k ro |  j | | | <qo qo Wt d |  j d	 | d
 t	 É } t
 r„ t | d d d d d É| d <n t rˆ d | d <n  t j | ç  } | |  d <| j pg  } xR | D]J } | j r_t | j | j | j | j É | | j <q%| j | | j <q%W| S(   s‹    The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        s
   multipart/R)   Rï   Rê  R⁄   Rô  t   CONTENT_LENGTHt   fpRÁ   t   keep_blank_valuesR(   R=   t   newlines   
s   _cgi.FieldStorage(   s   REQUEST_METHODs   CONTENT_TYPERΩ  (   Rä  t   content_typet
   startswithRë  R©  Rú  RÁ   R]   R3  R©   t   py31RH   R  t   cgit   FieldStorageR[   t   filenameRî  t   fileRì   RÅ  Rm   (	   RI   RZ  Rí  Ra   Rm   t   safe_envR”   R   Rñ  (    (    s&   /home/lgardner/git/professor/bottle.pyRY  ∏  s2    	 
	c         CÄ  s   |  j  j É  S(   sÛ    The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. (   t   urlpartst   geturl(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRŸ   €  s    s   bottle.request.urlpartsc         CÄ  s’   |  j  } | j d É p' | j d d É } | j d É pE | j d É } | sß | j d d É } | j d É } | rß | | d k rä d	 n d
 k rß | d | 7} qß n  t |  j É } t | | | | j d É d É S(   sı    The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. t   HTTP_X_FORWARDED_PROTOs   wsgi.url_schemet   httpt   HTTP_X_FORWARDED_HOSTt	   HTTP_HOSTt   SERVER_NAMEs	   127.0.0.1t   SERVER_PORTt   80t   443t   :Rê  Rï   (   RÁ   Rœ   t   urlquotet   fullpatht   UrlSplitResult(   RI   t   envRÃ  t   hostt   portRä   (    (    s&   /home/lgardner/git/professor/bottle.pyR…  „  s    	!$c         CÄ  s   t  |  j |  j j d É É S(   s:    Request path including :attr:`script_name` (if present). R‹   (   R#   t   script_nameRä   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR’  ı  s    c         CÄ  s   |  j  j d d É S(   sh    The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. Rê  Rï   (   RÁ   Rœ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   query_string˙  s    c         CÄ  s4   |  j  j d d É j d É } | r0 d | d Sd S(   sÒ    The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. RO  Rï   R‹   (   RÁ   Rœ   RP  (   RI   R⁄  (    (    s&   /home/lgardner/git/professor/bottle.pyR⁄     s    i   c         CÄ  s<   |  j  j d d É } t | |  j | É \ |  d <|  d <d S(   s˜    Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        RO  R‹   R€   N(   RÁ   Rœ   R8  Rä   (   RI   t   shiftt   script(    (    s&   /home/lgardner/git/professor/bottle.pyR8  	  s    c         CÄ  s   t  |  j j d É p d É S(   sﬁ    The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. RΩ  iˇˇˇˇ(   Rà   RÁ   Rœ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¢    s    c         CÄ  s   |  j  j d d É j É  S(   sA    The Content-Type header as a lowercase-string (default: empty). Rô  Rï   (   RÁ   Rœ   Rõ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR¡    s    c         CÄ  s%   |  j  j d d É } | j É  d k S(   s…    True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). t   HTTP_X_REQUESTED_WITHRï   t   xmlhttprequest(   RÁ   Rœ   Rõ  (   RI   t   requested_with(    (    s&   /home/lgardner/git/professor/bottle.pyt   is_xhr  s    c         CÄ  s   |  j  S(   s9    Alias for :attr:`is_xhr`. "Ajax" is not the right term. (   R·  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   is_ajax'  s    c         CÄ  sK   t  |  j j d d É É } | r% | S|  j j d É } | rG | d f Sd S(   s´   HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. t   HTTP_AUTHORIZATIONRï   t   REMOTE_USERN(   t
   parse_authRÁ   Rœ   Rg   (   RI   t   basict   ruser(    (    s&   /home/lgardner/git/professor/bottle.pyt   auth,  s      
c         CÄ  sa   |  j  j d É } | r> g  | j d É D] } | j É  ^ q( S|  j  j d É } | r] | g Sg  S(   s(   A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. t   HTTP_X_FORWARDED_FORR·   t   REMOTE_ADDR(   RÁ   Rœ   R@  RP  (   RI   t   proxyt   ipt   remote(    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_route:  s
     &c         CÄ  s   |  j  } | r | d Sd S(   sg    The client IP as a string. Note that this information can be forged
            by malicious clients. i    N(   RÓ  Rg   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_addrE  s    	c         CÄ  s   t  |  j j É  É S(   sD    Return a new :class:`Request` with a shallow :attr:`environ` copy. (   t   RequestRÁ   t   copy(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÒ  L  s    c         CÄ  s   |  j  j | | É S(   N(   RÁ   Rœ   (   RI   Rm   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   P  s    c         CÄ  s   |  j  | S(   N(   RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __getitem__Q  s    c         CÄ  s   d |  | <|  j  | =d  S(   NRï   (   RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delitem__R  s   
 c         CÄ  s   t  |  j É S(   N(   Ru  RÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  S  s    c         CÄ  s   t  |  j É S(   N(   R}   RÁ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __len__T  s    c         CÄ  s   |  j  j É  S(   N(   RÁ   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   U  s    c         CÄ  s¢   |  j  j d É r! t d É Ç n  | |  j  | <d } | d k rI d } n- | d
 k r^ d } n | j d É rv d } n  x% | D] } |  j  j d | d É q} Wd S(   sA    Change an environ value and clear all caches that depend on it. s   bottle.request.readonlys$   The environ dictionary is read-only.s
   wsgi.inputR3  Rï  Rò  Ró  RZ  R†  Rê  R’   t   HTTP_RÅ  Rã  s   bottle.request.N(    (   s   bodys   formss   filess   paramss   posts   json(   s   querys   params(   s   headerss   cookies(   RÁ   Rœ   R“   R¬  R—   Rg   (   RI   Ra   Rm   t   todelete(    (    s&   /home/lgardner/git/professor/bottle.pyt   __setitem__V  s    			c         CÄ  s   d |  j  j |  j |  j f S(   Ns   <%s: %s %s>(   t	   __class__RK   R¥   RŸ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  i  s    c         CÄ  s]   y5 |  j  d | } t | d É r0 | j |  É S| SWn! t k
 rX t d | É Ç n Xd S(   s@    Search in self.environ for additional user defined attributes. s   bottle.request.ext.%sRl   s   Attribute %r not defined.N(   RÁ   R2   Rl   R“   RO   (   RI   Rì   t   var(    (    s&   /home/lgardner/git/professor/bottle.pyt   __getattr__l  s
    $c         CÄ  s4   | d k r t  j |  | | É S| |  j d | <d  S(   NRÁ   s   bottle.request.ext.%s(   t   objectt   __setattr__RÁ   (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸  t  s     N(8   RK   RL   Rp   t	   __slots__R≤  Rg   Rc   R_   R©   RÔ   RA  R•   R  Rä   R¥   RÅ  RÜ  Rã  Rè  R’   Rï  Ró  Rò  R†  Rß  R∞  R∫  Rú  R3  R±  Rﬂ   RY  RŸ   R…  R’  R€  R⁄  R8  R¢  R¡  R·  R‚  RË  RÓ  RÔ  RÒ  Rœ   RÚ  RÛ  Rp  RÙ  R∑   R˜  R  R˙  R¸  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÉ  ﬁ  sd   			
#	
									c         CÄ  s   |  j  É  j d d É S(   NRƒ   t   -(   t   titlet   replace(   R0   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hkey{  s    t   HeaderPropertyc           BÄ  s5   e  Z d e d  d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   Rï   c         CÄ  s=   | | |  _  |  _ | | |  _ |  _ d | j É  |  _ d  S(   Ns   Current value of the %r header.(   Rì   R
   t   readert   writerRˇ  Rp   (   RI   Rì   R  R  R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Ä  s    c         CÄ  sE   | d  k r |  S| j j |  j |  j É } |  j rA |  j | É S| S(   N(   Rg   RÅ  Rœ   Rì   R
   R  (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   Ö  s     c         CÄ  s   |  j  | É | j |  j <d  S(   N(   R  RÅ  Rì   (   RI   Ri   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRn   ä  s    c         CÄ  s   | j  |  j =d  S(   N(   RÅ  Rì   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyRo   ç  s    N(   RK   RL   Rg   Rá   Rc   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR    s   		t   BaseResponsec        
   BÄ  sä  e  Z d  Z d Z d Z i e d+ É d 6e d, É d 6Z d d- d- d Ñ Z d- d Ñ Z	 d Ñ  Z
 d Ñ  Z e d Ñ  É Z e d Ñ  É Z d Ñ  Z d Ñ  Z e e e d- d É Z [ [ e d Ñ  É Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d- d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z e d  Ñ  É Z e d É Z e d d! e ÉZ e d" d! d# Ñ  d$ d% Ñ  ÉZ  e d& d' Ñ É Z! d- d( Ñ Z" d) Ñ  Z# d* Ñ  Z$ RS(.   s∫   Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    i»   s   text/html; charset=UTF-8s   Content-TypeiÃ   R‚   s   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Md5s   Last-Modifiedi0  Rï   c         KÄ  sµ   d  |  _ i  |  _ | |  _ | p' |  j |  _ | r{ t | t É rQ | j É  } n  x' | D] \ } } |  j	 | | É qX Wn  | r± x- | j É  D] \ } } |  j	 | | É qé Wn  d  S(   N(
   Rg   t   _cookiest   _headersR3  t   default_statusR1  R>   R]   t   itemsR2  (   RI   R3  R1  RÅ  t   more_headersRì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ¨  s    			c         CÄ  sì   | p	 t  } t | t  É s! t Ç | É  } |  j | _ t d Ñ  |  j j É  DÉ É | _ |  j rè t É  | _ | j j	 |  j j
 d d É É n  | S(   s    Returns a copy of self. c         sÄ  s"   |  ] \ } } | | f Vq d  S(   N(    (   R√   t   kt   v(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>¿  s    R≠  Rï   (   R  t
   issubclasst   AssertionErrorR1  R]   R  R	  R  R*   RW  t   output(   RI   Rj   RÒ  (    (    s&   /home/lgardner/git/professor/bottle.pyRÒ  ∫  s    	"	"c         CÄ  s   t  |  j É S(   N(   Ru  R3  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ∆  s    c         CÄ  s&   t  |  j d É r" |  j j É  n  d  S(   NRJ   (   R2   R3  RJ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   …  s    c         CÄ  s   |  j  S(   s;    The HTTP status line as a string (e.g. ``404 Not Found``).(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   status_lineÕ  s    c         CÄ  s   |  j  S(   s/    The HTTP status code as an integer (e.g. 404).(   R~  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRr  “  s    c         CÄ  s∂   t  | t É r( | t j | É } } n= d | k rY | j É  } t | j É  d É } n t d É Ç d | k o| d k n sê t d É Ç n  | |  _ t | p© d | É |  _	 d  S(   Nt    i    s+   String status line without a reason phrase.id   iÁ  s   Status code out of range.s
   %d Unknown(
   R>   Rà   t   _HTTP_STATUS_LINESRœ   RP  R@  R£   R~  Rá   R  (   RI   R1  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _set_status◊  s     	c         CÄ  s   |  j  S(   N(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _get_status„  s    sQ   A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. c         CÄ  s   t  É  } |  j | _ | S(   sl    An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. (   t
   HeaderDictR  R]   (   RI   t   hdict(    (    s&   /home/lgardner/git/professor/bottle.pyRÅ  Ó  s    	c         CÄ  s   t  | É |  j k S(   N(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __contains__ˆ  s    c         CÄ  s   |  j  t | É =d  S(   N(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  ˜  s    c         CÄ  s   |  j  t | É d S(   Niˇˇˇˇ(   R  R  (   RI   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  ¯  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R  R  (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  ˘  s    c         CÄ  s    |  j  j t | É | g É d S(   s|    Return the value of a previously defined header. If there is no
            header with that name, return a default value. iˇˇˇˇ(   R  Rœ   R  (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRÜ  ˚  s    c         CÄ  s    t  | É g |  j t | É <d S(   sh    Create a new response header, replacing any previously defined
            headers with the same name. N(   Rá   R  R  (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_header   s    c         CÄ  s,   |  j  j t | É g  É j t | É É d S(   s=    Add an additional response header, not removing duplicates. N(   R  R¨   R  R   Rá   (   RI   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR2    s    c         CÄ  s   |  j  S(   sx    Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. (   R4  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iter_headers	  s    c   	      CÄ  s  g  } t  |  j j É  É } d |  j k rF | j d |  j g f É n  |  j |  j k rî |  j |  j } g  | D] } | d | k ro | ^ qo } n  | g  | D]% \ } } | D] } | | f ^ qÆ qû 7} |  j r	x3 |  j j É  D] } | j d | j	 É  f É q„ Wn  | S(   s.    WSGI conform list of (header, value) tuples. s   Content-Typei    s
   Set-Cookie(
   R[   R  R	  R   t   default_content_typeR~  t   bad_headersR  Râ  t   OutputString(	   RI   Rx  RÅ  R  t   hRì   t   valst   valRà  (    (    s&   /home/lgardner/git/professor/bottle.pyR4    s    ,6	 R  t   Expiresc         CÄ  s   t  j t |  É É S(   N(   R   t   utcfromtimestampt
   parse_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   !  s    R  c         CÄ  s
   t  |  É S(   N(   t	   http_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   "  s    s   UTF-8c         CÄ  s:   d |  j  k r6 |  j  j d É d j d É d j É  S| S(   sJ    Return the charset specified in the content-type header (default: utf8). s   charset=iˇˇˇˇRö  i    (   R¡  R@  RP  (   RI   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRq  $  s    'c         KÄ  sk  |  j  s t É  |  _  n  | r< t t | | f | É É } n t | t É sZ t d É Ç n  t | É d k r{ t d É Ç n  | |  j  | <x‹ | j	 É  D]Œ \ } } | d k r⁄ t | t
 É r⁄ | j | j d d } q⁄ n  | d k rFt | t t f É r
| j É  } n' t | t t f É r1t j | É } n  t j d | É } n  | |  j  | | j d	 d
 É <qï Wd S(   sπ   Create a new cookie or replace an old one. If the `secret` parameter is
            set, create a `Signed Cookie` (described below).

            :param name: the name of the cookie.
            :param value: the value of the cookie.
            :param secret: a signature key required for signed cookies.

            Additionally, this method accepts all RFC 2109 attributes that are
            supported by :class:`cookie.Morsel`, including:

            :param max_age: maximum age in seconds. (default: None)
            :param expires: a datetime object or UNIX timestamp. (default: None)
            :param domain: the domain that is allowed to read the cookie.
              (default: current domain)
            :param path: limits the cookie to a given path (default: current path)
            :param secure: limit the cookie to HTTPS connections (default: off).
            :param httponly: prevents client-side javascript to read this cookie
              (default: off, requires Python 2.6 or newer).

            If neither `expires` nor `max_age` is set (default), the cookie will
            expire at the end of the browser session (as soon as the browser
            window is closed).

            Signed cookies may store any pickle-able object and are
            cryptographically signed to prevent manipulation. Keep in mind that
            cookies are limited to 4kb in most browsers.

            Warning: Signed cookies are not encrypted (the client can still see
            the content) and not copy-protected (the client can restore an old
            cookie). The main intention is to make pickling and unpickling
            save, not to store secret information at client side.
        s)   Secret key missing for non-string Cookie.i   s   Cookie value to long.t   max_agei   i  t   expiress   %a, %d %b %Y %H:%M:%S GMTRƒ   R˛  N(   R  R*   R/   t   cookie_encodeR>   R?  RJ  R}   R£   R	  R   t   secondst   dayst   datedateR   t	   timetupleRà   Râ   t   timet   gmtimet   strftimeR   (   RI   Rì   Rm   Rç  RC  Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_cookie+  s(    !	 c         KÄ  s+   d | d <d | d <|  j  | d | ç d S(   sq    Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. iˇˇˇˇR$  i    R%  Rï   N(   R.  (   RI   Ra   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete_cookiec  s    

c         CÄ  sD   d } x7 |  j  D], \ } } | d | j É  | j É  f 7} q W| S(   NRï   s   %s: %s
(   R4  Rˇ  RP  (   RI   Rx  Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  j  s    $(   s   Content-Type(   s   Allows   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Types   Content-Md5s   Last-ModifiedN(%   RK   RL   Rp   R  R  R\   R  Rg   Rc   RÒ  Rp  RJ   R  R  Rr  R  R  R1  RÅ  R  RÛ  RÚ  R˜  RÜ  R  R2  R  R4  R  R¡  Rà   R¢  R%  Rq  R.  R/  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  ë  sN    														8	c         Ä  s_   |  r t  d É n  t j É  â  á  f d Ü  } á  f d Ü  } á  f d Ü  } t | | | d É S(   Ns3   local_property() is deprecated and will be removed.c         Ä  s/   y à  j  SWn t k
 r* t d É Ç n Xd  S(   Ns    Request context not initialized.(   R˘  RO   RÑ  (   RI   (   t   ls(    s&   /home/lgardner/git/professor/bottle.pyt   fgett  s     c         Ä  s   | à  _  d  S(   N(   R˘  (   RI   Rm   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fsetx  s    c         Ä  s
   à  `  d  S(   N(   R˘  (   RI   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fdely  s    s   Thread-local property(   RY   t	   threadingt   localR  (   Rì   R1  R2  R3  (    (   R0  s&   /home/lgardner/git/professor/bottle.pyt   local_propertyq  s     t   LocalRequestc           BÄ  s    e  Z d  Z e j Z e É  Z RS(   sT   A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). (   RK   RL   Rp   RÉ  Rc   Rg  R6  RÁ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR7  }  s   	t   LocalResponsec           BÄ  sD   e  Z d  Z e j Z e É  Z e É  Z e É  Z	 e É  Z
 e É  Z RS(   s+   A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    (   RK   RL   Rp   R  Rc   Rg  R6  R  R~  R  R  R3  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  á  s   					R9  c           BÄ  s#   e  Z d  d d d Ñ Z d Ñ  Z RS(   Rï   c         KÄ  s#   t  t |  É j | | | | ç d  S(   N(   t   superR9  Rc   (   RI   R3  R1  RÅ  R
  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ö  s    c         CÄ  s@   |  j  | _  |  j | _ |  j | _ |  j | _ |  j | _ d  S(   N(   R~  R  R  R  R3  (   RI   Rh  (    (    s&   /home/lgardner/git/professor/bottle.pyR˙   ù  s
    N(   RK   RL   Rg   Rc   R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR9  ô  s   R§   c           BÄ  s#   e  Z d  Z d d d d d Ñ Z RS(   iÙ  c         KÄ  s2   | |  _  | |  _ t t |  É j | | | ç d  S(   N(   t	   exceptiont	   tracebackR9  R§   Rc   (   RI   R1  R3  R:  R;  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ß  s    		N(   RK   RL   R  Rg   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR§   •  s   t   PluginErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR<  µ  s    R!  c           BÄ  s)   e  Z d  Z d Z e d Ñ Z d Ñ  Z RS(   R†  i   c         CÄ  s   | |  _  d  S(   N(   R   (   RI   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   º  s    c         Ä  s)   |  j  â à s à  Sá  á f d Ü  } | S(   Nc          Ä  sõ   y à  |  | é  } Wn t  k
 r/ t É  } n Xt | t É rX à | É } d t _ | St | t É ró t | j t É ró à | j É | _ d | _ n  | S(   Ns   application/json(   R§   R   R>   R]   Rh  R¡  R9  R3  (   R4   RR   t   rvt   json_response(   R   R   (    s&   /home/lgardner/git/professor/bottle.pyRP   ¬  s    	!(   R   (   RI   R   RA  RP   (    (   R   R   s&   /home/lgardner/git/professor/bottle.pyR˙   ø  s
    	 (   RK   RL   Rì   R  R   Rc   R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR!  ∏  s   R"  c           BÄ  s#   e  Z d  Z d Z d Z d Ñ  Z RS(   s   This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. Rb  i   c         CÄ  s{   | j  j d É } t | t t f É rT t | É d k rT t | d | d ç | É St | t É rs t | É | É S| Sd  S(   NRb  i   i    i   (   Rı   Rœ   R>   RZ   R[   R}   t   viewRá   (   RI   R   RA  RÜ   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙   ﬁ  s    '(   RK   RL   Rp   Rì   R  R˙   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR"  ÷  s   t   _ImportRedirectc           BÄ  s&   e  Z d  Ñ  Z d d Ñ Z d Ñ  Z RS(   c         CÄ  sv   | |  _  | |  _ t j j | t j | É É |  _ |  j j j	 i t
 d 6g  d 6g  d 6|  d 6É t j j |  É d S(   s@    Create a virtual package that redirects imports (see PEP 302). t   __file__t   __path__t   __all__t
   __loader__N(   Rì   t   impmaskR   t   modulesR¨   t   impt
   new_modulet   moduleRs   t   updateRA  t	   meta_pathR   (   RI   Rì   RE  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   Í  s    		!c         CÄ  s=   d | k r d  S| j  d d É d } | |  j k r9 d  S|  S(   Nt   .i   i    (   t   rsplitRì   (   RI   t   fullnameRä   t   packname(    (    s&   /home/lgardner/git/professor/bottle.pyt   find_moduleÛ  s      c         CÄ  s   | t  j k r t  j | S| j d d É d } |  j | } t | É t  j | } t  j | <t |  j | | É |  | _ | S(   NRL  i   (   R   RF  RM  RE  t
   __import__Ru   RI  RD  (   RI   RN  t   modnamet   realnameRI  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_module˘  s     
	N(   RK   RL   Rc   Rg   RP  RT  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR@  È  s   		t	   MultiDictc           BÄ  s
  e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 e rê d	 Ñ  Z d
 Ñ  Z d Ñ  Z e
 Z e Z e Z e Z n? d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d d d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z e Z e Z RS(   sÈ    This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    c         OÄ  s,   t  d Ñ  t  | | é  j É  DÉ É |  _  d  S(   Nc         sÄ  s$   |  ] \ } } | | g f Vq d  S(   N(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R	  (   RI   R4   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s    c         CÄ  s   t  |  j É S(   N(   R}   R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ    s    c         CÄ  s   t  |  j É S(   N(   Ru  R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp    s    c         CÄ  s   | |  j  k S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s    c         CÄ  s   |  j  | =d  S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ    s    c         CÄ  s   |  j  | d S(   Niˇˇˇˇ(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ    s    c         CÄ  s   |  j  | | É d  S(   N(   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜    s    c         CÄ  s   |  j  j É  S(   N(   R]   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR∑     s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s   |  ] } | d  Vq d S(   iˇˇˇˇN(    (   R√   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   Râ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRâ    s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s%   |  ] \ } } | | d  f Vq d S(   iˇˇˇˇN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>   s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR	     s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R√   R  t   vlR  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>"  s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRì  !  s    c         CÄ  s$   g  |  j  j É  D] } | d ^ q S(   Niˇˇˇˇ(   R]   Râ  (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRâ  )  s    c         CÄ  s0   g  |  j  j É  D] \ } } | | d f ^ q S(   Niˇˇˇˇ(   R]   R	  (   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  *  s    c         CÄ  s   |  j  j É  S(   N(   R]   t   iterkeys(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRW  +  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s   |  ] } | d  Vq d S(   iˇˇˇˇN(    (   R√   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>,  s    (   R]   t
   itervalues(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRX  ,  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s%   |  ] \ } } | | d  f Vq d S(   iˇˇˇˇN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>.  s    (   R]   t	   iteritems(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRY  -  s    c         CÄ  s   d Ñ  |  j  j É  DÉ S(   Nc         sÄ  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R√   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>0  s    (   R]   RY  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iterallitems/  s    c         CÄ  s9   g  |  j  j É  D]% \ } } | D] } | | f ^ q  q S(   N(   R]   RY  (   RI   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRì  1  s    iˇˇˇˇc         CÄ  sA   y) |  j  | | } | r$ | | É S| SWn t k
 r< n X| S(   s”   Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        (   R]   Rm  (   RI   Ra   R
   t   indexR˛   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   4  s    
c         CÄ  s    |  j  j | g  É j | É d S(   s5    Add a new value to the list of values for this key. N(   R]   R¨   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   E  s    c         CÄ  s   | g |  j  | <d S(   s1    Replace the list of values with a single value. N(   R]   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   I  s    c         CÄ  s   |  j  j | É p g  S(   s5    Return a (possibly empty) list of values for a key. (   R]   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   getallM  s    N(   RK   RL   Rp   Rc   RÙ  Rp  R  RÛ  RÚ  R˜  R∑   R  Râ  R	  Rì  RW  RX  RY  RZ  Rg   Rœ   R   R   R\  t   getonet   getlist(    (    (    s&   /home/lgardner/git/professor/bottle.pyRU    s<   																						Rä  c           BÄ  sP   e  Z d  Z d Z e Z d d Ñ Z d d Ñ Z d d d Ñ Z	 e
 É  d Ñ Z RS(   s©   This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. R=   c         CÄ  sd   t  | t É r7 |  j r7 | j d É j | p3 |  j É St  | t É r\ | j | pX |  j É S| Sd  S(   NR)   (   R>   R?   t   recode_unicodeR@   RE   t   input_encodingRA   (   RI   R0   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _fixd  s
    c         CÄ  sq   t  É  } | p |  j } | _ t | _ xB |  j É  D]4 \ } } | j |  j | | É |  j | | É É q5 W| S(   s™    Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. (   Rä  R`  Rq   R_  Rì  R   Ra  (   RI   R(   RÒ  RB   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRE   l  s    		,c         CÄ  s7   y |  j  |  | | É SWn t t f k
 r2 | SXd S(   s7    Return the value as a unicode string, or the default. N(   Ra  Rf  R“   (   RI   Rì   R
   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   getunicodew  s    c         CÄ  sG   | j  d É r4 | j d É r4 t t |  É j | É S|  j | d | ÉS(   Nt   __R
   (   R¬  RB  R9  Rä  R˙  Rb  (   RI   Rì   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  ~  s    N(   RK   RL   Rp   R`  R©   R_  Rg   Ra  RE   Rb  R?   R˙  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRä  V  s   R  c           BÄ  sn   e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d d	 d
 Ñ Z d Ñ  Z RS(   sz    A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. c         OÄ  s,   i  |  _  | s | r( |  j | | é  n  d  S(   N(   R]   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   â  s    	 c         CÄ  s   t  | É |  j k S(   N(   R  R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ç  s    c         CÄ  s   |  j  t | É =d  S(   N(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  é  s    c         CÄ  s   |  j  t | É d S(   Niˇˇˇˇ(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  è  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  ê  s    c         CÄ  s,   |  j  j t | É g  É j t | É É d  S(   N(   R]   R¨   R  R   Rá   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   ë  s    c         CÄ  s    t  | É g |  j t | É <d  S(   N(   Rá   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   ì  s    c         CÄ  s   |  j  j t | É É p g  S(   N(   R]   Rœ   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR\  î  s    iˇˇˇˇc         CÄ  s   t  j |  t | É | | É S(   N(   RU  Rœ   R  (   RI   Ra   R
   R[  (    (    s&   /home/lgardner/git/professor/bottle.pyRœ   ï  s    c         CÄ  sJ   xC g  | D] } t  | É ^ q
 D]" } | |  j k r  |  j | =q  q  Wd  S(   N(   R  R]   (   RI   t   namesR◊   Rì   (    (    s&   /home/lgardner/git/professor/bottle.pyt   filteró  s    &N(   RK   RL   Rp   Rc   R  RÛ  RÚ  R˜  R   R   R\  Rg   Rœ   Re  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  Ö  s   								RÖ  c           BÄ  sq   e  Z d  Z d Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z RS(   s    This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    Rô  RΩ  c         CÄ  s   | |  _  d  S(   N(   RÁ   (   RI   RÁ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ´  s    c         CÄ  s3   | j  d d É j É  } | |  j k r+ | Sd | S(   s6    Translate header field name to CGI/WSGI environ key. R˛  Rƒ   Rı  (   R   R„   t   cgikeys(   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _ekeyÆ  s    c         CÄ  s   |  j  j |  j | É | É S(   s:    Return the header value as is (may be bytes or unicode). (   RÁ   Rœ   Rg  (   RI   Ra   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt   rawµ  s    c         CÄ  s   t  |  j |  j | É d É S(   NR)   (   R©  RÁ   Rg  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  π  s    c         CÄ  s   t  d |  j É Ç d  S(   Ns   %s is read-only.(   RJ  R¯  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  º  s    c         CÄ  s   t  d |  j É Ç d  S(   Ns   %s is read-only.(   RJ  R¯  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  ø  s    c         cÄ  so   xh |  j  D]] } | d  d k r> | d j d d É j É  Vq
 | |  j k r
 | j d d É j É  Vq
 q
 Wd  S(   Ni   Rı  Rƒ   R˛  (   RÁ   R   Rˇ  Rf  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ¬  s
    c         CÄ  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   …  s    c         CÄ  s   t  |  j É  É S(   N(   R}   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ     s    c         CÄ  s   |  j  | É |  j k S(   N(   Rg  RÁ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  À  s    (   s   CONTENT_TYPEs   CONTENT_LENGTHN(   RK   RL   Rp   Rf  Rc   Rg  Rg   Rh  RÚ  R˜  RÛ  Rp  R∑   RÙ  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÖ  ù  s   
								RÛ   c           BÄ  s∫   e  Z d  Z d Z d e f d Ñ  É  YZ d Ñ  Z d Ñ  Z d e d Ñ Z	 d	 Ñ  Z
 d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   sH   A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    t   _metaR  t	   Namespacec           BÄ  sÜ   e  Z d  Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z	 d Ñ  Z
 d	 Ñ  Z d
 Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   c         CÄ  s   | |  _  | |  _ d  S(   N(   t   _configt   _prefix(   RI   Rı   t	   namespace(    (    s&   /home/lgardner/git/professor/bottle.pyRc   €  s    	c         CÄ  s    t  d É |  j |  j d | S(   Ns}   Accessing namespaces as dicts is discouraged. Only use flat item access: cfg["names"]["pace"]["key"] -> cfg["name.space.key"]RL  (   RY   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÚ  ﬂ  s    
c         CÄ  s   | |  j  |  j d | <d  S(   NRL  (   Rk  Rl  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  Â  s    c         CÄ  s   |  j  |  j d | =d  S(   NRL  (   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  Ë  s    c         cÄ  sZ   |  j  d } xF |  j D]; } | j d É \ } } } | |  j  k r | r | Vq q Wd  S(   NRL  (   Rl  Rk  t
   rpartition(   RI   t	   ns_prefixRa   t   nst   dotRì   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  Î  s
    c         CÄ  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR∑   Ú  s    c         CÄ  s   t  |  j É  É S(   N(   R}   R∑   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRÙ  Û  s    c         CÄ  s   |  j  d | |  j k S(   NRL  (   Rl  Rk  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  Ù  s    c         CÄ  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ı  s    c         CÄ  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __str__ˆ  s    c         CÄ  sÑ   t  d É | |  k rM | d j É  rM t j |  j |  j d | É |  | <n  | |  k rw | j d É rw t | É Ç n  |  j | É S(   Ns   Attribute access is deprecated.i    RL  Rc  (	   RY   t   isupperRÛ   Rj  Rk  Rl  R¬  RO   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  ˘  s    
'c         CÄ  sé   | d k r | |  j  | <d  St d É t t | É rE t d É Ç n  | |  k rÄ |  | rÄ t |  | |  j É rÄ t d É Ç n  | |  | <d  S(   NRk  Rl  s#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   s   _configs   _prefix(   Rs   RY   R2   R9   RO   R>   R¯  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸    s    
,c         CÄ  so   | |  k rk |  j  | É } t | |  j É rk | d } x. |  D]# } | j | É r> |  | | =q> q> Wqk n  d  S(   NRL  (   R—   R>   R¯  R¬  (   RI   Ra   R  Rù   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delattr__  s    
c         OÄ  s   t  d É |  j | | é  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1     s    
(   RK   RL   Rc   RÚ  R˜  RÛ  Rp  R∑   RÙ  R  R  Rr  R˙  R¸  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRj  Ÿ  s   														c         OÄ  sB   i  |  _  d Ñ  |  _ | s! | r> t d É |  j | | é  n  d  S(   Nc         SÄ  s   d  S(   N(   Rg   (   Rì   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    s-   Constructor does no longer accept parameters.(   Ri  R  RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s
    	
c         CÄ  sx   t  É  } | j | É x[ | j É  D]M } xD | j | É D]3 \ } } | d k rb | d | } n  | |  | <q9 Wq# W|  S(   s   Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        t   DEFAULTt   bottleRL  (   Ru  s   bottle(   R-   Ro  t   sectionsR	  (   RI   R∆  RÜ   t   sectionRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_config!  s    	Rï   c   	      CÄ  s  | | f g } xÒ | r| j  É  \ } } t | t É sR t d t | É É Ç n  x™ | j É  D]ú \ } } t | t É sì t d t | É É Ç n  | rß | d | n | } t | t É rÒ | j | | f É | r˚ |  j |  | É |  | <q˚ q_ | |  | <q_ Wq W|  S(   s‰    Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        s   Source is not a dict (r)s   Key is not a string (%r)RL  (	   R—   R>   R]   RJ  R˛   R	  R?  R   Rj  (	   RI   t   sourceRm  RÓ   t   stackRù   Ra   Rm   t   full_key(    (    s&   /home/lgardner/git/professor/bottle.pyRÙ   1  s    	c         OÄ  s{   d } | rC t  | d t É rC | d j d É d } | d } n  x1 t | | é  j É  D] \ } } | |  | | <qY Wd S(   s’    If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` Rï   i    RL  i   N(   R>   R?  RP  R]   R	  (   RI   R4   RR   Rù   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ  I  s    "c         CÄ  s!   | |  k r | |  | <n  |  | S(   N(    (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¨   T  s    c         CÄ  sç   t  | t É s( t d t | É É Ç n  |  j | d d Ñ  É | É } | |  k rf |  | | k rf d  S|  j | | É t j |  | | É d  S(   Ns   Key has type %r (not a string)Re  c         SÄ  s   |  S(   N(    (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]  s    (   R>   R?  RJ  R˛   t   meta_getR  R]   R˜  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR˜  Y  s    c         CÄ  s   t  j |  | É d  S(   N(   R]   RÛ  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ  c  s    c         CÄ  s   x |  D] } |  | =q Wd  S(   N(    (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   clearf  s    c         CÄ  s   |  j  j | i  É j | | É S(   s-    Return the value of a meta field for a key. (   Ri  Rœ   (   RI   Ra   t	   metafieldR
   (    (    s&   /home/lgardner/git/professor/bottle.pyR}  j  s    c         CÄ  s:   | |  j  j | i  É | <| |  k r6 |  | |  | <n  d S(   sq    Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. N(   Ri  R¨   (   RI   Ra   R  Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  n  s    c         CÄ  s   |  j  j | i  É j É  S(   s;    Return an iterable of meta field names defined for a key. (   Ri  Rœ   R∑   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   meta_listu  s    c         CÄ  sv   t  d É | |  k r? | d j É  r? |  j |  | É |  | <n  | |  k ri | j d É ri t | É Ç n  |  j | É S(   Ns   Attribute access is deprecated.i    Rc  (   RY   Rs  Rj  R¬  RO   Rœ   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR˙  z  s    
c         CÄ  sì   | |  j  k r" t j |  | | É St d É t t | É rJ t d É Ç n  | |  k rÖ |  | rÖ t |  | |  j É rÖ t d É Ç n  | |  | <d  S(   Ns#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   R˝  R]   R¸  RY   R2   RO   R>   Rj  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR¸  Ç  s    
,c         CÄ  so   | |  k rk |  j  | É } t | |  j É rk | d } x. |  D]# } | j | É r> |  | | =q> q> Wqk n  d  S(   NRL  (   R—   R>   Rj  R¬  (   RI   Ra   R  Rù   (    (    s&   /home/lgardner/git/professor/bottle.pyRt  å  s    
c         OÄ  s   t  d É |  j | | é  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ï  s    
(   s   _metas
   _on_changeN(   RK   RL   Rp   R˝  R9   Rj  Rc   Ry  Rq   RÙ   RJ  R¨   R˜  RÛ  R~  Rg   R}  R  RÄ  R˙  R¸  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÛ   œ  s$   A					
						
		t   AppStackc           BÄ  s#   e  Z d  Z d Ñ  Z d d Ñ Z RS(   s>    A stack-like list. Calling it returns the head of the stack. c         CÄ  s   |  d S(   s)    Return the current default application. iˇˇˇˇ(    (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   ü  s    c         CÄ  s,   t  | t É s t É  } n  |  j | É | S(   s1    Add a new :class:`Bottle` instance to the stack (   R>   R  R   (   RI   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   push£  s    N(   RK   RL   Rp   R1   Rg   RÇ  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÅ  ú  s   	Rt  c           BÄ  s   e  Z d d Ñ Z d Ñ  Z RS(   i   i@   c         CÄ  sS   | | |  _  |  _ x9 d D]1 } t | | É r t |  | t | | É É q q Wd  S(   Nt   filenoRJ   Ro  t	   readlinest   tellR¥  (   s   filenos   closes   reads	   readliness   tells   seek(   Ræ  t   buffer_sizeR2   Ru   Rh   (   RI   Ræ  RÜ  R`   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ≠  s     c         cÄ  s?   |  j  |  j } } x% t r: | | É } | s2 d  S| Vq Wd  S(   N(   RÜ  Ro  R©   (   RI   RØ  Ro  R¶  (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ≤  s    	 i   (   RK   RL   Rc   Rp  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt  ´  s   Rw  c           BÄ  s,   e  Z d  Z d d Ñ Z d Ñ  Z d Ñ  Z RS(   sä    This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). c         CÄ  s   | |  _  t | É |  _ d  S(   N(   t   iteratorR^   t   close_callbacks(   RI   Rá  RJ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   æ  s    	c         CÄ  s   t  |  j É S(   N(   Ru  Rá  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  ¬  s    c         CÄ  s   x |  j  D] } | É  q
 Wd  S(   N(   Rà  (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   ≈  s    N(   RK   RL   Rp   Rg   Rc   Rp  RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw  ∫  s   	R  c           BÄ  sP   e  Z d  Z d e d d Ñ Z d	 d	 e d Ñ Z d Ñ  Z d Ñ  Z	 d d Ñ Z RS(
   sf   This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    s   ./t   allc         CÄ  s1   t  |  _ | |  _ | |  _ g  |  _ i  |  _ d  S(   N(   t   opent   openert   baset	   cachemodeRä   t   cache(   RI   Rå  Rã  Rç  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ‘  s
    				c         CÄ  s˚   t  j j t  j j | p |  j É É } t  j j t  j j | t  j j | É É É } | t  j 7} | |  j k rÉ |  j j | É n  | r¨ t  j j | É r¨ t  j	 | É n  | d k rÀ |  j j | É n |  j j | | É |  j j É  t  j j | É S(   s   Add a new path to the list of search paths. Return False if the
            path does not exist.

            :param path: The new search path. Relative paths are turned into
                an absolute and normalized form. If the path looks like a file
                (not ending in `/`), the filename is stripped off.
            :param base: Path used to absolutize relative search paths.
                Defaults to :attr:`base` which defaults to ``os.getcwd()``.
            :param index: Position within the list of search paths. Defaults
                to last index (appends to the list).

            The `base` parameter makes it easy to reference files installed
            along with a python module or package::

                res.add_path('./resources/', __file__)
        N(   t   osRä   t   abspatht   dirnameRå  R»   t   sepR+  t   isdirt   makedirsRg   R   R)  Ré  R~  t   exists(   RI   Rä   Rå  R[  t   create(    (    s&   /home/lgardner/git/professor/bottle.pyt   add_pathﬁ  s    '-c         cÄ  sï   |  j  } xÑ | rê | j É  } t j  j | É s7 q n  xS t j | É D]B } t j  j | | É } t j  j | É rÑ | j | É qG | VqG Wq Wd S(   s:    Iterate over all existing files in all registered paths. N(   Rä   R—   Rè  Rì  t   listdirR»   R   (   RI   t   searchRä   Rì   t   full(    (    s&   /home/lgardner/git/professor/bottle.pyRp  ˝  s    
	  c         CÄ  s†   | |  j  k s t rï x[ |  j D]P } t j j | | É } t j j | É r |  j d k rk | |  j  | <n  | Sq W|  j d k rï d |  j  | <qï n  |  j  | S(   s˙    Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. Râ  t   found(   s   alls   foundN(   Ré  R±   Rä   Rè  R»   t   isfileRç  Rg   (   RI   Rì   Rä   t   fpath(    (    s&   /home/lgardner/git/professor/bottle.pyt   lookup	  s    t   rc         OÄ  sA   |  j  | É } | s( t d | É Ç n  |  j | d | | | éS(   s=    Find a resource and return a file object, or raise IOError. s   Resource %r not found.R∫   (   Rû  t   IOErrorRã  (   RI   Rì   R∫   R”   R.  t   fname(    (    s&   /home/lgardner/git/professor/bottle.pyRä  	  s     N(
   RK   RL   Rp   Rä  Rc   Rg   Rq   Ró  Rp  Rû  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR     s   
		Rî  c           BÄ  sb   e  Z d d  Ñ Z e d É Z e d d e d d ÉZ e d Ñ  É Z	 d d	 Ñ Z
 e d d
 Ñ Z RS(   c         CÄ  s=   | |  _  | |  _ | |  _ | r- t | É n t É  |  _ d S(   s    Wrapper for file uploads. N(   R«  Rì   t   raw_filenameR  RÅ  (   RI   t   fileobjRì   R∆  RÅ  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   "	  s    			s   Content-Types   Content-LengthR  R
   iˇˇˇˇc         CÄ  sº   |  j  } t | t É s- | j d d É } n  t d | É j d d É j d É } t j j | j	 d t j j
 É É } t j d d | É j É  } t j d d	 | É j d
 É } | d  pª d S(   s—   Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        R=   t   ignoret   NFKDt   ASCIIs   \s   [^a-zA-Z0-9-_.\s]Rï   s   [-\s]+R˛  s   .-iˇ   t   empty(   R¢  R>   R?   RE   R   R@   Rè  Rä   t   basenameR   Rí  RÄ   RÅ   RP  (   RI   R°  (    (    s&   /home/lgardner/git/professor/bottle.pyR∆  0	  s    
	$$i   i   c         CÄ  sa   |  j  j | j |  j  j É  } } } x$ | | É } | s? Pn  | | É q) W|  j  j | É d  S(   N(   R«  Ro  R   RÖ  R¥  (   RI   Ræ  t
   chunk_sizeRo  R   Rú   t   buf(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _copy_fileC	  s    & c         CÄ  s£   t  | t É rè t j j | É r< t j j | |  j É } n  | rd t j j | É rd t d É Ç n  t	 | d É è } |  j
 | | É Wd QXn |  j
 | | É d S(   sÃ   Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        s   File exists.t   wbN(   R>   R?  Rè  Rä   Rì  R»   R∆  Rï  R†  Rä  R´  (   RI   t   destinationt	   overwriteR©  Ræ  (    (    s&   /home/lgardner/git/professor/bottle.pyt   saveK	  s    	Ni   i   (   RK   RL   Rg   Rc   R  R¡  Rà   R¢  Rr   R∆  R´  Rq   RØ  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRî   	  s   iÙ  s   Unknown Error.c         CÄ  s   t  |  | É Ç d S(   s+    Aborts execution and causes a HTTP error. N(   R§   (   R`  t   text(    (    s&   /home/lgardner/git/professor/bottle.pyt   aborth	  s    c         CÄ  st   | s* t  j d É d k r! d n d } n  t j d t É } | | _ d | _ | j d t t  j	 |  É É | Ç d S(	   sd    Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. t   SERVER_PROTOCOLs   HTTP/1.1i/  i.  Rj   Rï   t   LocationN(
   R7  Rœ   Rh  RÒ  R9  R1  R3  R  R#   RŸ   (   RŸ   R`  Rd  (    (    s&   /home/lgardner/git/professor/bottle.pyt   redirectm	  s    $		i   c         cÄ  s[   |  j  | É xG | d k rV |  j t | | É É } | s> Pn  | t | É 8} | Vq Wd S(   sF    Yield chunks from a range in a file. No chunk is bigger than maxread.i    N(   R¥  Ro  R£  R}   (   Ræ  Rú   RA   R•  R¶  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _file_iter_rangey	  s     t   autos   UTF-8c         CÄ  s@  t  j j | É t  j } t  j j t  j j | |  j d É É É }  t É  } |  j | É sh t d d É St  j j	 |  É sé t  j j
 |  É rõ t d d É St  j |  t  j É sΩ t d d É S| d k rÙ t j |  É \ } } | rÙ | | d <qÙ n  | r:| d	  d
 k r-| r-d | k r-| d | 7} n  | | d <n  | rut  j j | t k r[|  n | É } d | | d <n  t  j |  É } | j | d <} t j d t j | j É É }	 |	 | d <t j j d É }
 |
 r˜t |
 j d É d j É  É }
 n  |
 d% k	 rD|
 t | j É k rDt j d t j É  É | d <t d d | ç St j d k rYd n t  |  d É } d | d <t j j d É } d t j k r3t! t" t j d | É É } | s¬t d d  É S| d \ } } d! | | d" | f | d# <t# | | É | d <| r t$ | | | | É } n  t | d d$ | çSt | | ç S(&   sŸ   Open a file in a safe way and return :exc:`HTTPResponse` with status
        code 200, 305, 403 or 404. The ``Content-Type``, ``Content-Encoding``,
        ``Content-Length`` and ``Last-Modified`` headers are set if possible.
        Special support for ``If-Modified-Since``, ``Range`` and ``HEAD``
        requests.

        :param filename: Name or path of the file to send.
        :param root: Root path for file lookups. Should be an absolute directory
            path.
        :param mimetype: Defines the content-type header (default: guess from
            file extension)
        :param download: If True, ask the browser to open a `Save as...` dialog
            instead of opening the file with the associated program. You can
            specify a custom filename as a string. If not specified, the
            original filename is used (default: False).
        :param charset: The charset to use for files with a ``text/*``
            mime-type. (default: UTF-8)
    s   /\iì  s   Access denied.iî  s   File does not exist.s/   You do not have permission to access this file.R∂  s   Content-Encodingi   s   text/Rq  s   ; charset=%ss   Content-Types   attachment; filename="%s"s   Content-Dispositions   Content-Lengths   %a, %d %b %Y %H:%M:%S GMTs   Last-Modifiedt   HTTP_IF_MODIFIED_SINCERö  i    t   DateR1  i0  R›   Rï   t   rbRA   s   Accept-Rangest
   HTTP_RANGEi†  s   Requested Range Not Satisfiables   bytes %d-%d/%di   s   Content-RangeiŒ   N(%   Rè  Rä   Rê  Rí  R»   RP  R]   R¬  R§   Rï  Rú  t   accesst   R_OKt	   mimetypest
   guess_typeR®  R©   t   statt   st_sizeR+  R-  R,  t   st_mtimeR7  RÁ   Rœ   R"  R@  Rg   Rà   R9  R¥   Rä  R[   t   parse_range_headerRá   Rµ  (   R∆  t   roott   mimetypet   downloadRq  RÅ  R(   t   statsRª  t   lmt   imsR3  t   rangesRú   Rö   (    (    s&   /home/lgardner/git/professor/bottle.pyt   static_fileÉ	  sX    *	& "$
"!$
 c         CÄ  s&   |  r t  j d É n  t |  É a d S(   sS    Change the debug level.
    There is only one debug level supported at the moment.R
   N(   RT   t   simplefilterR  R±   (   R∫   (    (    s&   /home/lgardner/git/professor/bottle.pyt   debug‘	  s     c         CÄ  ss   t  |  t t f É r$ |  j É  }  n' t  |  t t f É rK t j |  É }  n  t  |  t É so t j	 d |  É }  n  |  S(   Ns   %a, %d %b %Y %H:%M:%S GMT(
   R>   R)  R   t   utctimetupleRà   Râ   R+  R,  R?  R-  (   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR#  €	  s    c         CÄ  se   y@ t  j j |  É } t j | d  d É | d p6 d t j SWn t t t t	 f k
 r` d SXd S(   sD    Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. i   i    i	   N(   i    (   t   emailt   utilst   parsedate_tzR+  t   mktimet   timezoneRJ  R£   t
   IndexErrort   OverflowErrorRg   (   R»  t   ts(    (    s&   /home/lgardner/git/professor/bottle.pyR"  ‰	  s
    .c         CÄ  sÑ   ye |  j  d d É \ } } | j É  d k rd t t j t | É É É j  d d É \ } } | | f SWn t t f k
 r d SXd S(   s]    Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or Nonei   RÊ  R”  N(	   R@  Rg   Rõ  R/   t   base64t	   b64decodeRC   R“   R£   (   R≠  R¥   R   t   usert   pwd(    (    s&   /home/lgardner/git/professor/bottle.pyRÂ  Ï	  s    -c         cÄ  s,  |  s |  d  d k r d Sg  |  d j  d É D]$ } d | k r/ | j  d d É ^ q/ } xÃ | D]ƒ \ } } y§ | sò t d | t | É É | } } nB | s¥ t | É | } } n& t | É t t | É d | É } } d | k o¸ | k  o¸ | k n r| | f Vn  Wq` t k
 r#q` Xq` Wd S(   s~    Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.i   s   bytes=NR·   R˛  i   i    (   R@  R°  Rà   R£  R£   (   R≠  t   maxlenRü  R…  Rò   Rö   (    (    s&   /home/lgardner/git/professor/bottle.pyR¬  ˆ	  s     >#&'c         CÄ  sª   g  } xÆ |  j  d d É j d É D]ë } | s4 q" n  | j d d É } t | É d k rh | j d É n  t | d j  d d	 É É } t | d j  d d	 É É } | j | | f É q" W| S(
   NRö  t   &t   =i   i   Rï   i    t   +R  (   R   R@  R}   R   t
   urlunquote(   t   qsRü  t   pairt   nvRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRë  
  s    "  c         CÄ  s6   t  d Ñ  t |  | É DÉ É o5 t |  É t | É k S(   ss    Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. c         sÄ  s-   |  ]# \ } } | | k r! d  n d Vq d S(   i    i   N(    (   R√   R    t   y(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>
  s    (   t   sumt   zipR}   (   R4   Rü  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _lscmp
  s    c         CÄ  s^   t  j t j |  d É É } t  j t j t | É | É j É  É } t d É | t d É | S(   s>    Encode and sign a pickle-able object. Return a (byte) string iˇˇˇˇt   !RŒ   (   R÷  t	   b64encodet   pickleR   t   hmact   newRC   t   digest(   R   Ra   R¡   t   sig(    (    s&   /home/lgardner/git/professor/bottle.pyR&  
  s    'c         CÄ  sá   t  |  É }  t |  É rÉ |  j t  d É d É \ } } t | d t j t j t  | É | É j É  É É rÉ t	 j
 t j | É É Sn  d S(   s?    Verify and decode an encoded string. Return an object or None.RŒ   i   N(   RC   t   cookie_is_encodedR@  RÂ  R÷  RÁ  RÈ  RÍ  RÎ  RË  R   R◊  Rg   (   R   Ra   RÏ  R¡   (    (    s&   /home/lgardner/git/professor/bottle.pyRå   
  s    4c         CÄ  s+   t  |  j t d É É o' t d É |  k É S(   s9    Return True if the argument looks like a encoded cookie.RÊ  RŒ   (   R  R¬  RC   (   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRÌ  *
  s    c         CÄ  s@   |  j  d d É j  d d É j  d d É j  d d É j  d	 d
 É S(   s;    Escape HTML special characters ``&<>`` and quotes ``'"``. R€  s   &amp;t   <s   &lt;t   >s   &gt;t   "s   &quot;t   's   &#039;(   R   (   t   string(    (    s&   /home/lgardner/git/professor/bottle.pyRÄ  /
  s    *c         CÄ  s2   d t  |  É j d d É j d d É j d d É S(   s;    Escape and quote a string to be used as an HTTP attribute.s   "%s"s   
s   &#10;s   s   &#13;s   	s   &#9;(   RÄ  R   (   RÚ  (    (    s&   /home/lgardner/git/professor/bottle.pyt
   html_quote5
  s    c         cÄ  sß   d |  j  j d d É j d É } t |  É } t | d É t | d pK g  É } | d | t | d |  É 7} | Vx) | d | D] } | d | 7} | VqÜ Wd S(   sì   Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    R‹   Rc  i    i   s   /<%s>N(   RK   R   RQ  R   R}   RZ   (   Rf   Rä   t   spect   argct   arg(    (    s&   /home/lgardner/git/professor/bottle.pyRX  ;
  s    
"$ c   	      CÄ  s}  | d k r |  | f S| j  d É j d É } |  j  d É j d É } | re | d d k re g  } n  | rÑ | d d k rÑ g  } n  | d k r√ | t | É k r√ | |  } | | } | | } nh | d k  r| t | É k r| | } | | } | |  } n( | d k  rd n d } t d | É Ç d d j | É } d d j | É } | j d É rs| rs| d 7} n  | | f S(   sS   Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    i    R‹   Rï   RO  R€   s"   Cannot shift. Nothing left from %s(   RP  R@  R}   R  R»   RB  (	   R⁄  t	   path_infoR‹  t   pathlistt
   scriptlistt   movedRß  t   new_script_namet   new_path_info(    (    s&   /home/lgardner/git/professor/bottle.pyR8  O
  s.    	 
 	 	



 t   privates   Access deniedc         Ä  s   á  á á f d Ü  } | S(   se    Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. c         Ä  s   á á  á á f d Ü  } | S(   Nc          Ä  se   t  j p d \ } } | d  k s1 à  | | É rX t d à É } | j d d à É | Sà |  | é  S(   Nië  s   WWW-Authenticates   Basic realm="%s"(   NN(   R7  RË  Rg   R§   R2  (   R4   RR   Rÿ  t   passwordRF   (   t   checkRf   t   realmR∞  (    s&   /home/lgardner/git/professor/bottle.pyRP   r
  s    (    (   Rf   RP   (   Rˇ  R   R∞  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  q
  s    (    (   Rˇ  R   R∞  R0  (    (   Rˇ  R   R∞  s&   /home/lgardner/git/professor/bottle.pyt
   auth_basicn
  s    	c         Ä  s+   t  j t t à  É É á  f d Ü  É } | S(   sA    Return a callable that relays calls to the current default app. c          Ä  s   t  t É  à  É |  | é  S(   N(   Rh   RÔ   (   R4   RR   (   Rì   (    s&   /home/lgardner/git/professor/bottle.pyRP   Ç
  s    (   RM   t   wrapsRh   R  (   Rì   RP   (    (   Rì   s&   /home/lgardner/git/professor/bottle.pyt   make_default_app_wrapperÄ
  s    'RA  Rœ   RZ  R\  R^  RØ   RE  R/  R   RL  RV  t   ServerAdapterc           BÄ  s/   e  Z e Z d  d d Ñ Z d Ñ  Z d Ñ  Z RS(   s	   127.0.0.1iê  c         KÄ  s%   | |  _  | |  _ t | É |  _ d  S(   N(   RC  Rÿ  Rà   RŸ  (   RI   Rÿ  RŸ  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   †
  s    		c         CÄ  s   d  S(   N(    (   RI   R_  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  •
  s    c         CÄ  sU   d j  g  |  j j É  D]" \ } } d | t | É f ^ q É } d |  j j | f S(   Ns   , s   %s=%ss   %s(%s)(   R»   RC  R	  RÊ   R¯  RK   (   RI   R  R  R”   (    (    s&   /home/lgardner/git/professor/bottle.pyR  ®
  s    A(   RK   RL   Rq   t   quietRc   RN  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  û
  s   	t	   CGIServerc           BÄ  s   e  Z e Z d  Ñ  Z RS(   c         Ä  s3   d d l  m } á  f d Ü  } | É  j | É d  S(   Niˇˇˇˇ(   t
   CGIHandlerc         Ä  s   |  j  d d É à  |  | É S(   NR€   Rï   (   R¨   (   RÁ   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyt   fixed_environ±
  s    (   t   wsgiref.handlersR  RN  (   RI   R_  R  R  (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN  Ø
  s    (   RK   RL   R©   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  ≠
  s   t   FlupFCGIServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  sN   d d  l  } |  j j d |  j |  j f É | j j j | |  j ç j É  d  S(   Niˇˇˇˇt   bindAddress(	   t   flup.server.fcgiRC  R¨   Rÿ  RŸ  t   servert   fcgit
   WSGIServerRN  (   RI   R_  t   flup(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ∏
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR
  ∑
  s   t   WSGIRefServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         Ä  s   d d l  m â  m } d d l  m } d d  l â d à  f á  á f d Ü  É  Y} à j j d | É } à j j d | É } d à j k rƒ t | d	 É à j	 k rƒ d
 | f á f d Ü  É  Y} qƒ n  | à j à j
 | | | É } | j É  d  S(   Niˇˇˇˇ(   t   WSGIRequestHandlerR  (   t   make_servert   FixedHandlerc           Ä  s#   e  Z d  Ñ  Z á  á f d Ü  Z RS(   c         SÄ  s   |  j  d S(   Ni    (   t   client_address(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   address_string≈
  s    c          Ä  s   à j  s à  j |  | é  Sd  S(   N(   R  t   log_request(   R”   t   kw(   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  «
  s    	(   RK   RL   R  R  (    (   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  ƒ
  s   	t   handler_classt   server_classR”  t   address_familyt
   server_clsc           Ä  s   e  Z à  j Z RS(    (   RK   RL   t   AF_INET6R  (    (   t   socket(    s&   /home/lgardner/git/professor/bottle.pyR  –
  s   (   t   wsgiref.simple_serverR  R  R  R  RC  Rœ   Rÿ  Rh   t   AF_INETRŸ  t   serve_forever(   RI   RÔ   R  R  R  t   handler_clsR  t   srv(    (   R  RI   R  s&   /home/lgardner/git/professor/bottle.pyRN  ø
  s    "(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  æ
  s   t   CherryPyServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s÷   d d l  m } |  j |  j f |  j d <| |  j d <|  j j d É } | r[ |  j d =n  |  j j d É } | rÄ |  j d =n  | j |  j ç  } | r§ | | _ n  | r∂ | | _ n  z | j	 É  Wd  | j
 É  Xd  S(   Niˇˇˇˇ(   t
   wsgiservert	   bind_addrt   wsgi_appt   certfilet   keyfile(   t   cherrypyR%  Rÿ  RŸ  RC  Rœ   t   CherryPyWSGIServert   ssl_certificatet   ssl_private_keyRò   t   stop(   RI   R_  R%  R(  R)  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  ÿ
  s"    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR$  ◊
  s   t   WaitressServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s0   d d l  m } | | d |  j d |  j Éd  S(   Niˇˇˇˇ(   t   serveRÿ  RŸ  (   t   waitressR0  Rÿ  RŸ  (   RI   R_  R0  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ò
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR/  
  s   t   PasteServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  se   d d l  m } d d l m } | | d |  j É} | j | d |  j d t |  j É |  j	 çd  S(   Niˇˇˇˇ(   t
   httpserver(   t   TransLoggert   setup_console_handlerRÿ  RŸ  (
   t   pasteR3  t   paste.transloggerR4  R  R0  Rÿ  Rá   RŸ  RC  (   RI   R_  R3  R4  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  ˜
  s
    !(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR2  ˆ
  s   t   MeinheldServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s:   d d l  m } | j |  j |  j f É | j | É d  S(   Niˇˇˇˇ(   R  (   t   meinheldR  t   listenRÿ  RŸ  RN  (   RI   R_  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN     s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  ˇ
  s   t   FapwsServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   sA    Extremely fast webserver using libev. See http://www.fapws.org/ c         Ä  s÷   d d  l  j } d d l m } m } |  j } t | j d É d k rV t | É } n  | j	 |  j
 | É d t j k rô |  j rô t d É t d É n  | j | É á  f d Ü  } | j d	 | f É | j É  d  S(
   Niˇˇˇˇ(   Rå  Rı   i˛ˇˇˇgöôôôôôŸ?t   BOTTLE_CHILDs3   WARNING: Auto-reloading does not work with Fapws3.
s/            (Fapws3 breaks python thread support)
c         Ä  s   t  |  d <à  |  | É S(   Ns   wsgi.multiprocess(   Rq   (   RÁ   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRÔ     s    
Rï   (   t   fapws._evwsgit   _evwsgit   fapwsRå  Rı   RŸ  Râ   t   SERVER_IDENTRá   Rò   Rÿ  Rè  RÁ   R  t   _stderrt   set_base_modulet   wsgi_cbRN  (   RI   R_  t   evwsgiRå  Rı   RŸ  RÔ   (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN    s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR;    s   t   TornadoServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s<    The super hyped asynchronous server by facebook. Untested. c         CÄ  s~   d d  l  } d d  l } d d  l } | j j | É } | j j | É } | j d |  j d |  j	 É | j
 j j É  j É  d  S(   NiˇˇˇˇRŸ  t   address(   t   tornado.wsgit   tornado.httpservert   tornado.ioloopRÇ  t   WSGIContainerR3  t
   HTTPServerR:  RŸ  Rÿ  t   ioloopt   IOLoopt   instanceRò   (   RI   R_  t   tornadot	   containerR  (    (    s&   /home/lgardner/git/professor/bottle.pyRN    s
    $(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRE    s   t   AppEngineServerc           BÄ  s   e  Z d  Z e Z d Ñ  Z RS(   s     Adapter for Google App Engine. c         Ä  sa   d d l  m â t j j d É } | rP t | d É rP á  á f d Ü  | _ n  à j à  É d  S(   Niˇˇˇˇ(   t   utilR   t   mainc           Ä  s   à j  à  É S(   N(   t   run_wsgi_app(    (   R_  RR  (    s&   /home/lgardner/git/professor/bottle.pyR!   /  s    (   t   google.appengine.ext.webappRR  R   RF  Rœ   R2   RS  RT  (   RI   R_  RI  (    (   R_  RR  s&   /home/lgardner/git/professor/bottle.pyRN  )  s
    (   RK   RL   Rp   R©   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRQ  &  s   t   TwistedServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  sß   d d l  m } m } d d l m } d d l m } | É  } | j É  | j d d | j	 É | j
 | j | | | É É } | j |  j | d |  j É| j É  d  S(   Niˇˇˇˇ(   R  RÇ  (   t
   ThreadPool(   t   reactort   aftert   shutdownt	   interface(   t   twisted.webR  RÇ  t   twisted.python.threadpoolRW  t   twisted.internetRX  Rò   t   addSystemEventTriggerR.  t   Sitet   WSGIResourcet	   listenTCPRŸ  Rÿ  RN  (   RI   R_  R  RÇ  RW  RX  t   thread_poolt   factory(    (    s&   /home/lgardner/git/professor/bottle.pyRN  5  s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRV  3  s   t   DieselServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  s3   d d l  m } | | d |  j É} | j É  d  S(   Niˇˇˇˇ(   t   WSGIApplicationRŸ  (   t   diesel.protocols.wsgiRf  RŸ  RN  (   RI   R_  Rf  RÔ   (    (    s&   /home/lgardner/git/professor/bottle.pyRN  C  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRe  A  s   t   GeventServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s„    Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    c         Ä  sı   d d l  m } m } m } t t j É  | j É sI d } t | É Ç n  |  j j d d  É sg | } n  |  j
 rv d  n d |  j d <|  j |  j f } | j | | |  j ç â  d t j k rÁ d d  l } | j | j á  f d Ü  É n  à  j É  d  S(	   Niˇˇˇˇ(   RÇ  t   pywsgiR5  s9   Bottle requires gevent.monkey.patch_all() (before import)t   fastR
   t   logR<  c         Ä  s
   à  j  É  S(   N(   R.  (   R0   Rÿ   (   R  (    s&   /home/lgardner/git/professor/bottle.pyR!   [  s    (   R   RÇ  Ri  R5  R>   R4  RÑ  RC  R—   Rg   R  Rÿ  RŸ  R  Rè  RÁ   t   signalt   SIGINTR!  (   RI   R_  RÇ  Ri  R5  R¡   RF  Rl  (    (   R  s&   /home/lgardner/git/professor/bottle.pyRN  P  s     	(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRh  I  s   t   GeventSocketIOServerc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  sB   d d l  m } |  j |  j f } | j | | |  j ç j É  d  S(   Niˇˇˇˇ(   R  (   t   socketioR  Rÿ  RŸ  t   SocketIOServerRC  R!  (   RI   R_  R  RF  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  `  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRn  _  s   t   GunicornServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s?    Untested. See http://gunicorn.org/configure.html for options. c         Ä  ss   d d l  m } i d |  j t |  j É f d 6â  à  j |  j É d | f á  á f d Ü  É  Y} | É  j É  d  S(   Niˇˇˇˇ(   t   Applications   %s:%dRg  t   GunicornApplicationc           Ä  s&   e  Z á  f d  Ü  Z á f d Ü  Z RS(   c         Ä  s   à  S(   N(    (   RI   t   parsert   optsR”   (   Rı   (    s&   /home/lgardner/git/professor/bottle.pyt   inito  s    c         Ä  s   à  S(   N(    (   RI   (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRW  r  s    (   RK   RL   Rv  RW  (    (   Rı   R_  (    s&   /home/lgardner/git/professor/bottle.pyRs  n  s   (   t   gunicorn.app.baseRr  Rÿ  Rà   RŸ  RJ  RC  RN  (   RI   R_  Rr  Rs  (    (   Rı   R_  s&   /home/lgardner/git/professor/bottle.pyRN  h  s
    #(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRq  f  s   t   EventletServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s
    Untested c         CÄ  sÄ   d d l  m } m } y0 | j | |  j |  j f É | d |  j ÉWn3 t k
 r{ | j | |  j |  j f É | É n Xd  S(   Niˇˇˇˇ(   RÇ  R:  t
   log_output(   t   eventletRÇ  R:  R  Rÿ  RŸ  R  RJ  (   RI   R_  RÇ  R:  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  z  s    !(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx  x  s   t   RocketServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s    Untested. c         CÄ  sC   d d l  m } | |  j |  j f d i | d 6É } | j É  d  S(   Niˇˇˇˇ(   t   RocketRÇ  R'  (   t   rocketR|  Rÿ  RŸ  Rò   (   RI   R_  R|  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  Ü  s    %(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{  Ñ  s   t   BjoernServerc           BÄ  s   e  Z d  Z d Ñ  Z RS(   s?    Fast server written in C: https://github.com/jonashaag/bjoern c         CÄ  s*   d d l  m } | | |  j |  j É d  S(   Niˇˇˇˇ(   RN  (   t   bjoernRN  Rÿ  RŸ  (   RI   R_  RN  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  é  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR~  å  s   t
   AutoServerc           BÄ  s,   e  Z d  Z e e e e e g Z d Ñ  Z	 RS(   s    Untested. c         CÄ  sR   xK |  j  D]@ } y& | |  j |  j |  j ç j | É SWq
 t k
 rI q
 Xq
 Wd  S(   N(   t   adaptersRÿ  RŸ  RC  RN  R   (   RI   R_  t   sa(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ñ  s
    &(
   RK   RL   Rp   R/  R2  RV  R$  R  RÅ  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÄ  ì  s   Rƒ  R  R1  R*  R6  t   fapws3RO  t   gaet   twistedt   dieselR9  t   gunicornRz  t   geventSocketIOR}  R  c         KÄ  s∏   d |  k r |  j  d d É n	 |  d f \ } }  | t j k rL t | É n  |  s] t j | S|  j É  r} t t j | |  É S| j  d É d } t j | | | <t d | |  f | É S(   sˇ   Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    R”  i   RL  i    s   %s.%sN(   R@  Rg   R   RF  RQ  t   isalnumRh   t   eval(   Rµ   Rm  RI  t   package_name(    (    s&   /home/lgardner/git/professor/bottle.pyRW  Ω  s    0   c         CÄ  sX   t  t a } z0 t j É  } t |  É } t | É r8 | S| SWd t j | É | a Xd S(   sﬁ    Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. N(   R©   t   NORUNt   default_appRÇ  RW  RI  R+  (   Rµ   t   nr_oldRπ  R=  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_app—  s    s	   127.0.0.1iê  c	         KÄ  sÿ  t  r
 d S| r~t j j d É r~z1yd }
 t j d d d d É \ } }
 t j | É x· t j j	 |
 É r=t
 j g t
 j } t j j É  } d | d <|
 | d <t j | d	 | É} x3 | j É  d k rÔ t j |
 d É t j | É qΩ W| j É  d
 k r] t j j	 |
 É r$t j |
 É n  t
 j | j É  É q] q] WWn t k
 rRn XWd t j j	 |
 É ryt j |
 É n  Xd Sy·| d k	 röt | É n  |  p¶t É  }  t |  t É r«t |  É }  n  t |  É sÊt d |  É Ç n  x! | pÚg  D] } |  j | É qÛW| t k r(t j | É } n  t | t É rFt  | É } n  t | t! É rp| d | d | |	 ç } n  t | t" É sít d | É Ç n  | j# pû| | _# | j# sÓt$ d t% t& | É f É t$ d | j' | j( f É t$ d É n  | rQt j j d É }
 t) |
 | É } | è | j* |  É Wd QX| j+ d k r^t
 j d
 É q^n | j* |  É Wnr t k
 rrnb t, t- f k
 rãÇ  nI | söÇ  n  t. | d | É s∂t/ É  n  t j | É t
 j d
 É n Xd S(   sº   Start a server instance. This method blocks until the server terminates.

        :param app: WSGI application or target string supported by
               :func:`load_app`. (default: :func:`default_app`)
        :param server: Server adapter to use. See :data:`server_names` keys
               for valid names or pass a :class:`ServerAdapter` subclass.
               (default: `wsgiref`)
        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on
               all interfaces including the external one. (default: 127.0.0.1)
        :param port: Server port to bind to. Values below 1024 require root
               privileges. (default: 8080)
        :param reloader: Start auto-reloading server? (default: False)
        :param interval: Auto-reloader interval in seconds (default: 1)
        :param quiet: Suppress output to stdout and stderr? (default: False)
        :param options: Options passed to the server adapter.
     NR<  Rù   s   bottle.t   suffixs   .lockt   truet   BOTTLE_LOCKFILER◊  i   s   Application is not callable: %rRÿ  RŸ  s!   Unknown or unsupported server: %rs,   Bottle v%s server starting up (using %s)...
s   Listening on http://%s:%d/
s   Hit Ctrl-C to quit.

t   reloadR  (0   Rå  Rè  RÁ   Rœ   Rg   t   tempfilet   mkstempRJ   Rä   Rï  R   t
   executablet   argvRÒ  t
   subprocesst   Popent   pollt   utimeR+  t   sleept   unlinkt   exitRj  t   _debugRç  R>   R?  Rè  RI  R£   R   t   server_namesRW  R˛   R  R  RA  t   __version__RÊ   Rÿ  RŸ  t   FileCheckerThreadRN  R1  Rk  Rl  Rh   R   (   RÔ   R  Rÿ  RŸ  t   intervalt   reloaderR  RÒ   RÃ  RS  t   lockfilet   fdR”   RÁ   RÇ   R  t   bgcheck(    (    s&   /home/lgardner/git/professor/bottle.pyRN  ﬂ  sà      

  	 
R¢  c           BÄ  s2   e  Z d  Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z RS(   sw    Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. c         CÄ  s0   t  j j |  É | | |  _ |  _ d  |  _ d  S(   N(   R4  t   ThreadRc   R•  R£  Rg   R1  (   RI   R•  R£  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ?  s    c         CÄ  s[  t  j j } d Ñ  } t É  } xq t t j j É  É D]Z } t | d d É } | d d k ri | d  } n  | r4 | | É r4 | | É | | <q4 q4 Wx¬ |  j	 sV| |  j
 É s‘ | |  j
 É t j É  |  j d k  rÍ d	 |  _	 t j É  n  xV t | j É  É D]B \ } } | | É s(| | É | k r˝ d
 |  _	 t j É  Pq˝ q˝ Wt j |  j É qï Wd  S(   Nc         SÄ  s   t  j |  É j S(   N(   Rè  Rø  R¡  (   Rä   (    (    s&   /home/lgardner/git/professor/bottle.pyR!   G  s    RA  Rï   i¸ˇˇˇs   .pyos   .pyciˇˇˇˇi   RØ   Rì  (   s   .pyos   .pyc(   Rè  Rä   Rï  R]   R[   R   RF  Râ  Rh   R1  R•  R+  R£  t   threadt   interrupt_mainR	  Rú  (   RI   Rï  t   mtimeRò  RI  Rä   t   lmtime(    (    s&   /home/lgardner/git/professor/bottle.pyRN  E  s(    		  &		
c         CÄ  s   |  j  É  d  S(   N(   Rò   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   __enter__[  s    c         CÄ  s8   |  j  s d |  _  n  |  j É  | d  k	 o7 t | t É S(   NRû  (   R1  R»   Rg   R  Rj  (   RI   t   exc_typet   exc_valt   exc_tb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __exit__^  s    	 
(   RK   RL   Rp   Rc   RN  R≠  R±  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR¢  ;  s
   			t   TemplateErrorc           BÄ  s   e  Z d  Ñ  Z RS(   c         CÄ  s   t  j |  d | É d  S(   NiÙ  (   R§   Rc   (   RI   RW   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   m  s    (   RK   RL   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR≤  l  s   t   BaseTemplatec           BÄ  st   e  Z d  Z d d d d g Z i  Z i  Z d d g  d d Ñ Z e g  d Ñ É Z	 e d Ñ  É Z
 d	 Ñ  Z d
 Ñ  Z RS(   s2    Base class and minimal API for template adapters t   tplt   htmlt   thtmlt   stplR=   c         KÄ  s+  | |  _  t | d É r$ | j É  n | |  _ t | d É rE | j n d |  _ g  | D] } t j j | É ^ qU |  _	 | |  _
 |  j j É  |  _ |  j j | É |  j rÙ |  j  rÙ |  j |  j  |  j	 É |  _ |  j sÙ t d t | É É Ç qÙ n  |  j r|  j rt d É Ç n  |  j |  j ç  d S(   s=   Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        Ro  R∆  s   Template %s not found.s   No template specified.N(   Rì   R2   Ro  Rz  R∆  Rg   Rè  Rä   Rê  Rû  R(   t   settingsRÒ  RJ  Rô  R≤  RÊ   R˘   (   RI   Rz  Rì   Rû  R(   R∏  R    (    (    s&   /home/lgardner/git/professor/bottle.pyRc   w  s    	$!(		c         CÄ  s  | s t  d É d g } n  t j j | É rZ t j j | É rZ t  d É t j j | É Sx± | D]© } t j j | É t j } t j j t j j | | É É } | j | É s∂ qa n  t j j | É rÃ | Sx; |  j	 D]0 } t j j d | | f É r÷ d | | f Sq÷ Wqa Wd S(   s{    Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. s2   The template lookup path list should not be empty.RL  s,   Absolute template path names are deprecated.s   %s.%sN(
   RY   Rè  Rä   t   isabsRú  Rê  Rí  R»   R¬  t
   extensions(   Rj   Rì   Rû  t   spathR°  t   ext(    (    s&   /home/lgardner/git/professor/bottle.pyRô  ë  s     
$
!  c         GÄ  s;   | r, |  j  j É  |  _  | d |  j  | <n |  j  | Sd S(   sB    This reads or sets the global settings stored in class.settings. i    N(   R∏  RÒ  (   Rj   Ra   R”   (    (    s&   /home/lgardner/git/professor/bottle.pyt   global_config¶  s    c         KÄ  s
   t  Ç d S(   sô    Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        N(   t   NotImplementedError(   RI   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   Ø  s    c         OÄ  s
   t  Ç d S(   sF   Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        N(   Ræ  (   RI   R”   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   render∂  s    N(   RK   RL   Rp   R∫  R∏  t   defaultsRg   Rc   t   classmethodRô  RΩ  R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR≥  q  s   		t   MakoTemplatec           BÄ  s   e  Z d  Ñ  Z d Ñ  Z RS(   c         KÄ  s¥   d d l  m } d d l m } | j i |  j d 6É | j d t t É É | d |  j	 | ç } |  j
 râ | |  j
 d | | ç|  _ n' | d |  j d	 |  j d | | ç |  _ d  S(
   Niˇˇˇˇ(   t   Template(   t   TemplateLookupR`  t   format_exceptionst   directoriesRû  t   uriR∆  (   t   mako.templateR√  t   mako.lookupRƒ  RJ  R(   R¨   R  R±   Rû  Rz  R¥  Rì   R∆  (   RI   RC  R√  Rƒ  Rû  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   ¡  s    	c         OÄ  sJ   x | D] } | j  | É q W|  j j É  } | j  | É |  j j | ç  S(   N(   RJ  R¿  RÒ  R¥  Rø  (   RI   R”   R.  t   dictargt	   _defaults(    (    s&   /home/lgardner/git/professor/bottle.pyRø  Ã  s
     (   RK   RL   R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR¬  ¿  s   	t   CheetahTemplatec           BÄ  s   e  Z d  Ñ  Z d Ñ  Z RS(   c         KÄ  s~   d d l  m } t j É  |  _ i  |  j _ |  j j g | d <|  j rb | d |  j | ç |  _ n | d |  j | ç |  _ d  S(   Niˇˇˇˇ(   R√  t
   searchListRz  R«  (	   t   Cheetah.TemplateR√  R4  R5  R  t   varsRz  R¥  R∆  (   RI   RC  R√  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   ‘  s    	c         OÄ  sj   x | D] } | j  | É q W|  j j j  |  j É |  j j j  | É t |  j É } |  j j j É  | S(   N(   RJ  R  Rœ  R¿  Rá   R¥  R~  (   RI   R”   R.  R   Rx  (    (    s&   /home/lgardner/git/professor/bottle.pyRø  ﬁ  s     (   RK   RL   R˘   Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÃ  ”  s   	
t   Jinja2Templatec           BÄ  s,   e  Z d d i  d  Ñ Z d Ñ  Z d Ñ  Z RS(   c         KÄ  s„   d d l  m } m } d | k r1 t d É Ç n  | d | |  j É | ç |  _ | rk |  j j j | É n  | rá |  j j j | É n  | r£ |  j j	 j | É n  |  j
 r« |  j j |  j
 É |  _ n |  j j |  j É |  _ d  S(   Niˇˇˇˇ(   t   Environmentt   FunctionLoaderRù   ss   The keyword argument `prefix` has been removed. Use the full jinja2 environment name line_statement_prefix instead.t   loader(   t   jinja2R—  R“  RÑ  R”  R◊  Rí   RJ  t   testst   globalsRz  t   from_stringR¥  t   get_templateR∆  (   RI   Rí   R’  R÷  R.  R—  R“  (    (    s&   /home/lgardner/git/professor/bottle.pyR˘   Ë  s       	c         OÄ  sJ   x | D] } | j  | É q W|  j j É  } | j  | É |  j j | ç  S(   N(   RJ  R¿  RÒ  R¥  Rø  (   RI   R”   R.  R   RÀ  (    (    s&   /home/lgardner/git/professor/bottle.pyRø  ˆ  s
     c         CÄ  sQ   |  j  | |  j É } | s d  St | d É è } | j É  j |  j É SWd  QXd  S(   NRπ  (   Rô  Rû  Rä  Ro  RE   R(   (   RI   Rì   R°  Rÿ   (    (    s&   /home/lgardner/git/professor/bottle.pyR”  ¸  s
     N(   RK   RL   Rg   R˘   Rø  R”  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR–  Á  s   	t   SimpleTemplatec           BÄ  sb   e  Z e e d d  Ñ Z e d Ñ  É Z e d Ñ  É Z d d Ñ Z	 d d Ñ Z
 d Ñ  Z d Ñ  Z RS(   c         Ä  sh   i  |  _  |  j â  á  f d Ü  |  _ á  á f d Ü  |  _ | |  _ | rd |  j |  j |  _ |  _ n  d  S(   Nc         Ä  s   t  |  à  É S(   N(   R/   (   R    (   RB   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    c         Ä  s   à t  |  à  É É S(   N(   R/   (   R    (   RB   t   escape_func(    s&   /home/lgardner/git/professor/bottle.pyR!   	  s    (   Ré  R(   t   _strt   _escapet   syntax(   RI   R⁄  t   noescapeR›  RR   (    (   RB   R⁄  s&   /home/lgardner/git/professor/bottle.pyR˘     s    			c         CÄ  s   t  |  j |  j p d d É S(   Ns   <string>R<   (   RÆ   R`  R∆  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   co  s    c         CÄ  sª   |  j  } | s9 t |  j d É è } | j É  } Wd  QXn  y t | É d } } Wn1 t k
 rÉ t d É t | d É d } } n Xt | d | d |  j É} | j	 É  } | j
 |  _
 | S(   NRπ  R=   s;   Template encodings other than utf8 are no longer supported.R)   R(   R›  (   Rz  Rä  R∆  Ro  R/   Rf  RY   t
   StplParserR›  t	   translateR(   (   RI   Rz  Rÿ   R(   Rt  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR`    s    	
c         KÄ  s0   | d  k r t d t É n  | | f | d <d  S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?t   _rebase(   Rg   RY   R©   (   RI   t   _envR‘   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyR‚  "  s    
c         KÄ  sÑ   | d  k r t d t É n  | j É  } | j | É | |  j k ri |  j d | d |  j É |  j | <n  |  j | j | d | É S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?Rì   Rû  t   _stdout(	   Rg   RY   R©   RÒ  RJ  Ré  R¯  Rû  t   execute(   RI   R„  R‘   R.  R◊  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _include(  s    
%c         CÄ  s  |  j  j É  } | j | É | j i
 | d 6| j d 6t j |  j | É d 6t j |  j | É d 6d  d 6|  j	 d 6|  j
 d 6| j d 6| j d	 6| j d
 6É t |  j | É | j d É r˝ | j d É \ } } d j | É | d <| 2|  j | | | ç S| S(   NR‰  t
   _printlistt   includet   rebaseR‚  R€  R‹  Rœ   R¨   t   definedRï   Rå  (   R¿  RÒ  RJ  t   extendRM   R  RÊ  R‚  Rg   R€  R‹  Rœ   R¨   R  Rä  Rﬂ  R—   R»   (   RI   R‰  R.  R◊  t   subtplt   rargs(    (    s&   /home/lgardner/git/professor/bottle.pyRÂ  2  s    c         OÄ  sT   i  } g  } x | D] } | j  | É q W| j  | É |  j | | É d j | É S(   sA    Render the template using keyword arguments as local variables. Rï   (   RJ  RÂ  R»   (   RI   R”   R.  R◊  R   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRø  B  s      N(   RK   RL   RÄ  Rq   Rg   R˘   Rr   Rﬂ  R`  R‚  RÊ  RÂ  Rø  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRŸ    s   	
	t   StplSyntaxErrorc           BÄ  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRÓ  K  s    R‡  c           BÄ  sÒ   e  Z d  Z i  Z d Z e j d d É Z e d 7Z e d 7Z e d 7Z e d 7Z e d 7Z e d	 7Z e d
 7Z d Z d e Z d Z d d d Ñ Z
 d Ñ  Z d Ñ  Z e e e É Z d Ñ  Z d Ñ  Z d Ñ  Z d Ñ  Z d d Ñ Z d Ñ  Z RS(   s    Parser for stpl templates. sà   ((?m)[urbURB]?(?:''(?!')|""(?!")|'{6}|"{6}|'(?:[^\\']|\\.)+?'|"(?:[^\\"]|\\.)+?"|'{3}(?:[^\\]|\\.|\n)+?'{3}|"{3}(?:[^\\]|\\.|\n)+?"{3}))s   |\nRï   s   |(#.*)s   |([\[\{\(])s   |([\]\}\)])sW   |^([ \t]*(?:if|for|while|with|try|def|class)\b)|^([ \t]*(?:elif|else|except|finally)\b)s?   |((?:^|;)[ \t]*end[ \t]*(?=(?:%(block_close)s[ \t]*)?\r?$|;|#))s   |(%(block_close)s[ \t]*(?=$))s   |(\r?\n)s8   (?m)^[ 	]*(\\?)((%(line_start)s)|(%(block_start)s))(%%?)s2   %%(inline_start)s((?:%s|[^'"
]*?)+)%%(inline_end)ss   <% %> % {{ }}R=   c         CÄ  sv   t  | | É | |  _ |  _ |  j | p. |  j É g  g  |  _ |  _ d \ |  _ |  _ d \ |  _	 |  _
 d |  _ d  S(   Ni   i    (   i   i    (   i    i    (   R/   Rz  R(   t
   set_syntaxt   default_syntaxt   code_buffert   text_buffert   linenoRú   t   indentt
   indent_modt   paren_depth(   RI   Rz  R›  R(   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   n  s    c         CÄ  s   |  j  S(   s=    Tokens as a space separated string (default: <% %> % {{ }}) (   t   _syntax(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_syntaxv  s    c         CÄ  sŒ   | |  _  | j É  |  _ | |  j k r´ d } t t j |  j É } t t | j É  | É É } |  j	 |  j
 |  j f } g  | D] } t j | | É ^ q| } | |  j | <n  |  j | \ |  _ |  _ |  _ d  S(   Ns:   block_start block_close line_start inline_start inline_end(   R˜  R@  t   _tokenst	   _re_cachet   mapRÄ   R´   R]   R‰  t	   _re_splitt   _re_tokt   _re_inlRÆ   t   re_splitt   re_tokt   re_inl(   RI   R›  Rd  t   etokenst   pattern_varst   patternsRÇ   (    (    s&   /home/lgardner/git/professor/bottle.pyRÔ  z  s    	&c         CÄ  sÓ  |  j  r t d É Ç n  xüt rπ|  j j |  j |  j  É } | rµ|  j |  j  |  j  | j É  !} |  j j | É |  j  | j	 É  7_  | j
 d É r
|  j |  j  j d É \ } } } |  j j | j
 d É | j
 d É | | É |  j  t | | É d 7_  q n | j
 d É rât d É |  j |  j  j d É \ } } } |  j j | j
 d É | | É |  j  t | | É d 7_  q n  |  j É  |  j d t | j
 d É É É q Pq W|  j j |  j |  j  É |  j É  d	 j |  j É S(
   Ns   Parser is a one time instance.i   s   
i   i   s#   Escape code lines with a backslash.t	   multilinei   Rï   (   Rú   RÑ  R©   Rˇ  Rô  Rz  Rò   RÚ  R   Rö   R~   R®  R}   RY   t
   flush_textt	   read_codeR  R»   RÒ  (   RI   R   R∞  t   lineRí  Rƒ   (    (    s&   /home/lgardner/git/professor/bottle.pyR·  à  s2    	 	 ".
"!
"
c      	   CÄ  su  d \ } } xbt  rp|  j j |  j |  j É } | sw | |  j |  j 7} t |  j É |  _ |  j | j É  | É d  S| |  j |  j |  j | j É  !7} |  j | j	 É  7_ | j
 É  \	 } } } } }	 }
 } } } | sÏ |  j d k r|	 s¯ |
 r| |	 p|
 7} q n  | r!| | 7} q | r[| } | rm| j É  j |  j d É rmt } qmq | r}|  j d 7_ | | 7} q | r±|  j d k r§|  j d 8_ n  | | 7} q |	 rŸ|	 d } |  _ |  j d 7_ q |
 rÚ|
 d } |  _ q | r
|  j d 8_ q | r,| rt } qm| | 7} q |  j | j É  | É |  j d 7_ d \ } } |  _ | s Pq q Wd  S(   NRï   i    i   iˇˇˇˇ(   Rï   Rï   (   Rï   Rï   i    (   R©   R   Rô  Rz  Rú   R}   t
   write_codeRP  Rò   Rö   Rô   Rˆ  RB  R˘  Rq   Rı  RÙ  RÛ  (   RI   R  t	   code_linet   commentR   R€  t   _comt   _pot   _pct   _blk1t   _blk2t   _endt   _cendt   _nl(    (    s&   /home/lgardner/git/professor/bottle.pyR  ¢  sV    	$'!" 	c   	      CÄ  s‘  d j  |  j É } |  j 2| s# d  Sg  d d d |  j } } } x≤ |  j j | É D]û } | | | j É  !| j É  } } | r¨ | j | j  t t	 | j
 t É É É É n  | j d É rŒ | d c | 7<n  | j |  j | j d É j É  É É qU W| t | É k  rî| | } | j
 t É } | d j d É rJ| d d	  | d <n( | d j d
 É rr| d d  | d <n  | j | j  t t	 | É É É n  d d j  | É } |  j | j d É d 7_ |  j | É d  S(   NRï   i    s   \
s     s   
iˇˇˇˇi   s   \\
i˝ˇˇˇs   \\
i¸ˇˇˇs   _printlist((%s,))s   , (   R»   RÚ  RÙ  R  Ró   Rò   Rö   R   R˚  RÊ   t
   splitlinesR©   RB  t   process_inlineR~   RP  R}   RÛ  t   countR	  (	   RI   R∞  t   partst   post   nlR   Rù   t   linesR`  (    (    s&   /home/lgardner/git/professor/bottle.pyR  —  s.      + )
  "c         CÄ  s$   | d d k r d | d Sd | S(   Ni    RÊ  s   _str(%s)i   s   _escape(%s)(    (   RI   t   chunk(    (    s&   /home/lgardner/git/professor/bottle.pyR  Ê  s     c         CÄ  sX   |  j  | | É \ } } d |  j |  j } | | j É  | d 7} |  j j | É d  S(   Ns     s   
(   t   fix_backward_compatibilityRÙ  Rı  RQ  RÒ  R   (   RI   R  R  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  Í  s    c         CÄ  s7  | j  É  j d  d É } | rë | d d k rë t d É t | É d k rT d | f St | É d k rz d t | É | f Sd	 t | É | f Sn  |  j d k r-| j  É  r-d
 | k r-t j d | É } | r-t d É | j	 d É } |  j
 j |  j É j | É |  _
 | |  _ | | j d
 d É f Sn  | | f S(   Ni   i    RË  RÈ  s2   The include and rebase keywords are functions now.i   s   _printlist([base])s   _=%s(%r)s   _=%s(%r, %s)t   codings   #.*coding[:=]\s*([-\w.]+)s4   PEP263 encoding strings in templates are deprecated.s   coding*(   s   includes   rebase(   RP  R@  Rg   RY   R}   RZ   RÛ  RÄ   Rû   R~   Rz  R@   R(   RE   R   (   RI   R  R  R  R   RB   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s"    
 
 (
!	N(   RK   RL   Rp   R˙  R˝  R   R˛  R¸  R  Rg   Rc   R¯  RÔ  R  R›  R·  R  R  R  R	  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR‡  N  s0   







				/		c          OÄ  se  |  r |  d n d } | j d t É } | j d t É } t | É | f } | t k s^ t r| j d i  É } t | | É r¶ | t | <| rt | j | ç  qqd | k s÷ d | k s÷ d | k s÷ d | k rı | d	 | d
 | | ç t | <q| d | d
 | | ç t | <n  t | s2t	 d d | É n  x |  d D] } | j
 | É q=Wt | j | É S(   sÍ   
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    i    t   template_adaptert   template_lookupt   template_settingss   
t   {t   %t   $Rz  Rû  Rì   iÙ  s   Template (%s) not foundi   N(   Rg   R—   RŸ  t   TEMPLATE_PATHt   idt	   TEMPLATESR±   R>   R˘   R±  RJ  Rø  (   R”   R.  R¥  t   adapterRû  t   tplidR∏  R   (    (    s&   /home/lgardner/git/professor/bottle.pyRb    s$    
 0
 R  c         Ä  s   á  á f d Ü  } | S(   s…   Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    c         Ä  s(   t  j à  É á á  á f d Ü  É } | S(   Nc          Ä  sg   à |  | é  } t  | t t f É rJ à  j É  } | j | É t à | ç S| d  k rc t à à  É S| S(   N(   R>   R]   R9   RÒ  RJ  Rb  Rg   (   R”   R.  t   resultt   tplvars(   R¿  Rf   t   tpl_name(    s&   /home/lgardner/git/professor/bottle.pyRP   +  s    (   RM   R  (   Rf   RP   (   R¿  R+  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  *  s    $
(    (   R+  R¿  R0  (    (   R¿  R+  s&   /home/lgardner/git/professor/bottle.pyR?     s    
s   ./s   ./views/s	   ../views/s   I'm a teapoti¢  s   Unprocessable Entityi¶  s   Precondition Requiredi¨  s   Too Many Requestsi≠  s   Request Header Fields Too LargeiØ  s   Network Authentication Requirediˇ  c         cÄ  s+   |  ]! \ } } | d  | | f f Vq d S(   s   %d %sN(    (   R√   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>S  s    sÇ  
%%try:
    %%from %s import DEBUG, HTTP_CODES, request, touni
    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
    <html>
        <head>
            <title>Error: {{e.status}}</title>
            <style type="text/css">
              html {background-color: #eee; font-family: sans;}
              body {background-color: #fff; border: 1px solid #ddd;
                    padding: 15px; margin: 15px;}
              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}
            </style>
        </head>
        <body>
            <h1>Error: {{e.status}}</h1>
            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>
               caused an error:</p>
            <pre>{{e.body}}</pre>
            %%if DEBUG and e.exception:
              <h2>Exception:</h2>
              <pre>{{repr(e.exception)}}</pre>
            %%end
            %%if DEBUG and e.traceback:
              <h2>Traceback:</h2>
              <pre>{{e.traceback}}</pre>
            %%end
        </body>
    </html>
%%except ImportError:
    <b>ImportError:</b> Could not generate the error page. Please add bottle to
    the import path.
%%end
s
   bottle.exts   .exts	   bottle_%ss
   Bottle %s
s"   
Error: No application specified.
RL  Rv  t	   localhostR”  t   ]s   []Rÿ  RŸ  R  R§  RÒ   RÃ  (  Rp   t
   __future__R    t
   __author__R°  t   __license__RK   t   optparseR   t   _cmd_parsert
   add_optiont   _optt
   parse_argst   _cmd_optionst	   _cmd_argsR  R¬  t   gevent.monkeyR   t   monkeyt	   patch_allR÷  Rƒ  t   email.utilsRŒ  RM   RÈ  RG  R:  RΩ  Rè  RÄ   Rò  R   Rî  R4  R+  RT   R   R   R)  R   R   R;  R   R   t   inspectR   t   unicodedataR   t
   simplejsonR   R   R   R.   R   R†  t   django.utils.simplejsont   version_infot   pyR  t   py25R√  R   R   R   R"   R‰  RA  R†  t   http.clientt   clientt   httplibt   _threadR©  t   urllib.parseR#   R$   R÷  R%   R&   R‘  R'   Rﬁ  R  t   http.cookiesR*   t   collectionsR+   R9   RË  t   ioR,   t   configparserR-   Rá   R?  R?   Rù  RI  R˚  R6   R5   t   urlparset   urllibt   Cookiet   cPickleR7   R8   R¡   RU   RV   t   UserDictR:   RA   Rä  RÆ   RC   R/   R©  RG   RH   RN   Rq   RY   R^   R˚  R_   Rr   Rt   Rm  Rv   Rw   Rx   Ry   Rz   R{   RÉ   RÑ   RÌ   R  RÉ  R  R  R  Rg   R6  R7  R8  R  t   ResponseR9  R§   R<  R!  R"  R@  RU  Rä  R  RÖ  R]   RÛ   R[   RÅ  Rt  Rw  R  Rî  R±  R¥  Rµ  R   R©   RÃ  R#  R"  RÂ  R¬  Rë  RÂ  R&  Rå  RÌ  RÄ  RÛ  RX  R8  R  R  RA  Rœ   RZ  R\  R^  RØ   RE  R/  R   RL  RŸ   R  R  R
  R  R$  R/  R2  R8  R;  RE  RQ  RV  Re  Rh  Rn  Rq  Rx  R{  R~  RÄ  R†  RW  Rè  Rü  RN  R®  R¢  R≤  R≥  R¬  RÃ  R–  RŸ  RÓ  R‡  Rb  t   mako_templatet   cheetah_templatet   jinja2_templateR?  t	   mako_viewt   cheetah_viewt   jinja2_viewR$  R&  R±   Rå  t	   responsest
   HTTP_CODESR	  R  Rc  R7  Rh  R5  RÔ   Rç  RÇ  RI  Rº  t   optR”   Rt  t   versionRû  t
   print_helpRä   R)  RF  R¨   Rg  Rÿ  RŸ  t   rfindRM  RP  Rà   Rì  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyt   <module>   s  	 ¿   		.	"									»wˇ °ˇ û	‡
$I/2ÕVH
Q				
				
					
	


		Z1OH¥			





$
		
(	

*(#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

#cd $parent_path
echo $parent_path

python2.7 ./run.py -p 8081
#!/bin/python
import bottle as bottle
from bottle import *

#Static
@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static/')

#Template
@route('/')
def main():
    return template('index.tpl')

@post("/GID")
def post_gid():
    USER_IN = request.query.get("gid") or ""
    print("x")
    return template('accounts.tpl', USER_IN=USER_IN)
@get("/GID")
def get_gid():
        #print(request.query.get("User0"))
        #print(request.query.get('User1'))
        print(request.query.get('confirmed'))
run(host='localhost', port=8081, debug=True)
