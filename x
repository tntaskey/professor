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
�
��Xc           @�  s�  d  Z  d d l m Z d Z d Z d Z e d k r5d d l m Z e d d	 � Z	 e	 j
 Z e d
 d d d d �e d d d d d d �e d d d d d d �e d d d d d d �e d d d d d �e d d d d d  �e	 j �  \ Z Z e j oe j j d! � r2d d" l Z e j j �  n  n  d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l  Z  d d" l! Z! d d" l" Z" d d" l# Z# d d" l$ Z$ d d" l% Z% d d# l& m' Z( m& Z& m) Z) d d$ l" m* Z* d d% l+ m, Z, m- Z- d d& l. m/ Z/ d d' l0 m1 Z1 y d d( l2 m3 Z4 m5 Z6 Wn| e7 k
 r�y d d( l8 m3 Z4 m5 Z6 WnN e7 k
 r�y d d( l9 m3 Z4 m5 Z6 Wn  e7 k
 r�d) �  Z4 e4 Z6 n Xn Xn Xe! j: Z; e; d* d+ d+ f k Z< e; d, d- d+ f k  Z= d* d. d+ f e; k oLd* d, d+ f k  n Z> d/ �  Z? y" e! j@ jA e! jB jA f \ ZC ZD Wn# eE k
 r�d0 �  ZC d1 �  ZD n Xe< r�d d" lF jG ZH d d" lI ZJ d d2 lK mL ZL mM ZN d d3 lK mO ZO mP ZQ mR ZS e jT eS d4 d5 �ZS d d6 lU mV ZV d d7 lW mX ZY d d" lZ ZZ d d8 l[ m\ Z\ d d9 l] m^ Z^ e_ Z` e_ Za d: �  Zb d; �  Zc ed Ze d< �  Zf nd d" lH ZH d d" lJ ZJ d d2 lg mL ZL mM ZN d d3 lh mO ZO mP ZQ mR ZS d d6 li mV ZV d d= l me Ze d d" lj ZZ d d> lk mk Z\ d d? l^ ml Z^ e= rZd@ Zm e% jn em eo � d dA lp mY ZY dB �  Zq e_ Zr n d d7 lW mX ZY ea Za e6 Zb es et dC dD dE � � dF dG � Zu dF dH dI � Zv e< r�ev n eu Zw e> r�d dJ l[ mx Zx dK ex f dL �  �  YZy n  dM �  Zz e{ dN � Z| dO �  Z} dP e~ f dQ �  �  YZ dR e~ f dS �  �  YZ� dT e~ f dU �  �  YZ� dV e� f dW �  �  YZ� dX e� f dY �  �  YZ� dZ e� f d[ �  �  YZ� d\ e� f d] �  �  YZ� d^ e� f d_ �  �  YZ� d` e� f da �  �  YZ� db �  Z� dc e~ f dd �  �  YZ� de e~ f df �  �  YZ� dg e~ f dh �  �  YZ� di e~ f dj �  �  YZ� dk �  Z� dl e~ f dm �  �  YZ� dn e~ f do �  �  YZ� e� dp � Z� dq e� f dr �  �  YZ� ds e� f dt �  �  YZ� e� Z� e� Z� du e� e� f dv �  �  YZ� dw e� f dx �  �  YZ� dy e� f dz �  �  YZ� d{ e~ f d| �  �  YZ� d} e~ f d~ �  �  YZ� d e~ f d� �  �  YZ� d� eY f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� eY f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� d� d� � Z� e� d� � Z� d� d� d� � Z� d� e{ d� d� � Z� e� d� � Z� d� �  Z� d� �  Z� d� �  Z� d+ d� � Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d. d� � Z� d� d� d� � Z� d� �  Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� d� e~ f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� i e� d� 6e� d� 6e� d 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d! 6e� d� 6e� d� 6e� d� 6e� d� 6Z� d� �  Z� d� �  Z� e� Z� e� d d� d� d. e{ e{ e� e� d� �	 Z� d� e# j� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e� f d� �  �  YZ� d e� f d�  �  YZ� de� f d�  �  YZ� de� f d�  �  YZ� de� f d�  �  YZ� de~ f d	�  �  YZ� d
�  Z� e jT e� de� �Z� e jT e� de� �Z� e jT e� de� �Z� d�  Z� e jT e� de� �Z� e jT e� de� �Z� e jT e� de� �Z� dddg Z� i  Z� e{ a� e{ a� eH j� Z� de� d<de� d<de� d<de� d<de� d<de� d<e� d�  e� j� �  D� � Z� de Z� e� �  Z� e� �  Z� e# j� �  Z� e� �  Z Ze j�  e� e d k rdn e dd � jZe d k r�e e e	 f \ ZZZejrueC d!e � e! j	d+ � n  er�ej
�  eD d"� e! j	d. � n  e! jjd+ d#� e! jjd$e! jd � ejp�d%d� f \ ZZd&ek oejd'� ejd&� k  r-ejd&d. � \ ZZn  ejd(� Ze� ed+ d)ed*ee� d+ej d,ejd-ejd.ej� �n  d" S(/  s�  
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
i����(   t   with_statements   Marcel Hellkamps   0.12.9t   MITt   __main__(   t   OptionParsert   usages)   usage: %prog [options] package.module:apps	   --versiont   actiont
   store_truet   helps   show version number.s   -bs   --bindt   metavart   ADDRESSs   bind socket to ADDRESS.s   -ss   --servert   defaultt   wsgirefs   use SERVER as backend.s   -ps   --plugint   appends   install additional plugin/s.s   --debugs   start server in debug mode.s   --reloads   auto-reload on file changes.t   geventN(   t   datet   datetimet	   timedelta(   t   TemporaryFile(   t
   format_exct	   print_exc(   t
   getargspec(   t	   normalize(   t   dumpst   loadsc         C�  s   t  d � � d  S(   Ns/   JSON support requires Python 2.6 or simplejson.(   t   ImportError(   t   data(    (    s&   /home/lgardner/git/professor/bottle.pyt
   json_dumps6   s    i   i    i   i   i   c           C�  s   t  j �  d S(   Ni   (   t   syst   exc_info(    (    (    s&   /home/lgardner/git/professor/bottle.pyt   _eE   s    c         C�  s   t  j j |  � S(   N(   R   t   stdoutt   write(   t   x(    (    s&   /home/lgardner/git/professor/bottle.pyt   <lambda>L   s    c         C�  s   t  j j |  � S(   N(   R   t   stderrR   (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   M   s    (   t   urljoint   SplitResult(   t	   urlencodet   quotet   unquotet   encodingt   latin1(   t   SimpleCookie(   t   MutableMapping(   t   BytesIO(   t   ConfigParserc         C�  s   t  t |  � � S(   N(   t   json_ldst   touni(   t   s(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]   s    c         C�  s   t  |  d � S(   Nt   __call__(   t   hasattr(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ^   s    c          G�  s%   |  d |  d � j  |  d � � d  S(   Ni    i   i   (   t   with_traceback(   t   a(    (    s&   /home/lgardner/git/professor/bottle.pyt   _raise`   s    (   t   imap(   t   StringIO(   t   SafeConfigParsers?   Python 2.5 support may be dropped in future versions of Bottle.(   t	   DictMixinc         C�  s
   |  j  �  S(   N(   t   next(   t   it(    (    s&   /home/lgardner/git/professor/bottle.pyR:   o   s    s&   def _raise(*a): raise a[0], a[1], a[2]s   <py3fix>t   exect   utf8c         C�  s&   t  |  t � r |  j | � St |  � S(   N(   t
   isinstancet   unicodet   encodet   bytes(   R0   t   enc(    (    s&   /home/lgardner/git/professor/bottle.pyt   tobx   s    t   strictc         C�  s)   t  |  t � r |  j | | � St |  � S(   N(   R>   RA   t   decodeR?   (   R0   RB   t   err(    (    s&   /home/lgardner/git/professor/bottle.pyR/   z   s    (   t   TextIOWrappert   NCTextIOWrapperc           B�  s   e  Z d  �  Z RS(   c         C�  s   d  S(   N(    (   t   self(    (    s&   /home/lgardner/git/professor/bottle.pyt   close�   s    (   t   __name__t
   __module__RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRH   �   s   c         O�  s2   y t  j |  | | | � Wn t k
 r- n Xd  S(   N(   t	   functoolst   update_wrappert   AttributeError(   t   wrappert   wrappedR4   t   ka(    (    s&   /home/lgardner/git/professor/bottle.pyRN   �   s      c         C�  s   t  j |  t d d �d  S(   Nt
   stackleveli   (   t   warningst   warnt   DeprecationWarning(   t   messaget   hard(    (    s&   /home/lgardner/git/professor/bottle.pyt   depr�   s    c         C�  s:   t  |  t t t t f � r% t |  � S|  r2 |  g Sg  Sd  S(   N(   R>   t   tuplet   listt   sett   dict(   R   (    (    s&   /home/lgardner/git/professor/bottle.pyt   makelist�   s
     
 t   DictPropertyc           B�  sA   e  Z d  Z d e d � Z d �  Z d �  Z d �  Z d �  Z	 RS(   s=    Property that maps to a key in a local dict-like attribute. c         C�  s!   | | | |  _  |  _ |  _ d  S(   N(   t   attrt   keyt	   read_only(   RI   R`   Ra   Rb   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __init__�   s    c         C�  s9   t  j |  | d g  �| |  j p( | j |  _ |  _ |  S(   Nt   updated(   RM   RN   Ra   RK   t   getter(   RI   t   func(    (    s&   /home/lgardner/git/professor/bottle.pyR1   �   s    c         C�  sV   | d  k r |  S|  j t | |  j � } } | | k rN |  j | � | | <n  | | S(   N(   t   NoneRa   t   getattrR`   Re   (   RI   t   objt   clsRa   t   storage(    (    s&   /home/lgardner/git/professor/bottle.pyt   __get__�   s      c         C�  s5   |  j  r t d � � n  | t | |  j � |  j <d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   t   value(    (    s&   /home/lgardner/git/professor/bottle.pyt   __set__�   s    	 c         C�  s2   |  j  r t d � � n  t | |  j � |  j =d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   __delete__�   s    	 N(
   RK   RL   t   __doc__Rg   t   FalseRc   R1   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR_   �   s   			t   cached_propertyc           B�  s    e  Z d  Z d �  Z d �  Z RS(   s�    A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. c         C�  s   t  | d � |  _ | |  _ d  S(   NRp   (   Rh   Rp   Rf   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �   s    c         C�  s4   | d  k r |  S|  j | � } | j |  j j <| S(   N(   Rg   Rf   t   __dict__RK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �   s      (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRr   �   s   	t   lazy_attributec           B�  s    e  Z d  Z d �  Z d �  Z RS(   s4    A property that caches itself to the class object. c         C�  s#   t  j |  | d g  �| |  _ d  S(   NRd   (   RM   RN   Re   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �   s    c         C�  s&   |  j  | � } t | |  j | � | S(   N(   Re   t   setattrRK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �   s    (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt   �   s   	t   BottleExceptionc           B�  s   e  Z d  Z RS(   s-    A base class for exceptions used by bottle. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRv   �   s   t
   RouteErrorc           B�  s   e  Z d  Z RS(   s9    This is a base class for all routing related exceptions (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw   �   s   t
   RouteResetc           B�  s   e  Z d  Z RS(   sf    If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx   �   s   t   RouterUnknownModeErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRy   �   s    t   RouteSyntaxErrorc           B�  s   e  Z d  Z RS(   s@    The route parser found something not supported by this router. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRz   �   s   t   RouteBuildErrorc           B�  s   e  Z d  Z RS(   s    The route could not be built. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{   �   s   c         C�  s&   d |  k r |  St  j d d �  |  � S(   s^    Turn all capturing groups in a regular expression pattern into
        non-capturing groups. t   (s   (\\*)(\(\?P<[^>]+>|\((?!\?))c         S�  s7   t  |  j d � � d r& |  j d � S|  j d � d S(   Ni   i   i    s   (?:(   t   lent   group(   t   m(    (    s&   /home/lgardner/git/professor/bottle.pyR!   �   s    (   t   ret   sub(   t   p(    (    s&   /home/lgardner/git/professor/bottle.pyt   _re_flatten�   s     	t   Routerc           B�  st   e  Z d  Z d Z d Z d Z e d � Z d �  Z e	 j
 d � Z d �  Z d d � Z d	 �  Z d
 �  Z d �  Z RS(   sA   A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    s   [^/]+R�   ic   c         �  sz   g  �  _  i  �  _ i  �  _ i  �  _ i  �  _ i  �  _ | �  _ i �  f d �  d 6d �  d 6d �  d 6d �  d 6�  _ d  S(	   Nc         �  s   t  |  p �  j � d  d  f S(   N(   R�   t   default_patternRg   (   t   conf(   RI   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d t  d �  f S(   Ns   -?\d+c         S�  s   t  t |  � � S(   N(   t   strt   int(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d t  d �  f S(   Ns   -?[\d.]+c         S�  s   t  t |  � � S(   N(   R�   t   float(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d S(   Ns   .+?(   s   .+?NN(   Rg   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!      s    t   path(   t   rulest   _groupst   buildert   statict   dyna_routest   dyna_regexest   strict_ordert   filters(   RI   RD   (    (   RI   s&   /home/lgardner/git/professor/bottle.pyRc     s    							

c         C�  s   | |  j  | <d S(   s�    Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. N(   R�   (   RI   t   nameRf   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   add_filter"  s    s�   (\\*)(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)(?::((?:\\.|[^\\>]+)+)?)?)?>))c   	      c�  s?  d \ } } x� |  j  j | � D]� } | | | | j �  !7} | j �  } t | d � d r� | | j d � t | d � 7} | j �  } q n  | r� | d  d  f Vn  | d d  k r� | d d !n
 | d d !\ } } } | | p� d | p� d  f V| j �  d } } q W| t | � k s"| r;| | | d  d  f Vn  d  S(	   Ni    t    i   i   i   i   R
   (   i    R�   (   t   rule_syntaxt   finditert   startt   groupsR}   R~   t   endRg   (	   RI   t   rulet   offsett   prefixt   matcht   gR�   t   filtrR�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _itertokens-  s    !3c         �  s�  d } g  } d } g  �  g  } t  }	 x|  j | � D]\ }
 } } | rt }	 | d k rg |  j } n  |  j | | � \ } } } |
 s� | d | 7} d | }
 | d 7} n! | d |
 | f 7} | j |
 � | r� �  j |
 | f � n  | j |
 | p� t f � q4 |
 r4 | t j |
 � 7} | j d |
 f � q4 q4 W| |  j
 | <| r]| |  j
 | <n  |	 r�|  j r�|  j j | i  � | d f |  j | |  j | � <d Sy  t j d	 | � } | j � Wn- t j k
 r�t d
 | t �  f � � n X�  r�  � f d �  } n! | j r*� f d �  } n d } t | � } | | | | f } | | f |  j k r�t r�d } t j | | | f t � n  | |  j | |  j | | f <n@ |  j j | g  � j | � t |  j | � d |  j | | f <|  j | � d S(   s<    Add a new rule or replace the target for an existing rule. i    R�   R
   s   (?:%s)s   anon%di   s
   (?P<%s>%s)Ns   ^(%s)$s   Could not add Route: %s (%s)c         �  sh   � |  � j  �  } xO �  D]G \ } } y | | | � | | <Wq t k
 r_ t d d � � q Xq W| S(   Ni�  s   Path has wrong format.(   t	   groupdictt
   ValueErrort	   HTTPError(   R�   t   url_argsR�   t   wildcard_filter(   R�   t   re_match(    s&   /home/lgardner/git/professor/bottle.pyt   getargsh  s    c         �  s   �  |  � j  �  S(   N(   R�   (   R�   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyR�   q  s    s3   Route <%s %s> overwrites a previously defined route(   t   TrueR�   Rq   t   default_filterR�   R   R�   R�   t   escapeRg   R�   R�   R�   t
   setdefaultt   buildt   compileR�   t   errorRz   R   t
   groupindexR�   R�   t   DEBUGRT   RU   t   RuntimeWarningR�   R}   t   _compile(   RI   R�   t   methodt   targetR�   t   anonst   keyst   patternR�   t	   is_staticRa   t   modeR�   t   maskt	   in_filtert
   out_filtert
   re_patternR�   t   flatpatt
   whole_rulet   msg(    (   R�   R�   s&   /home/lgardner/git/professor/bottle.pyt   add>  sf     
   	!$c         C�  s�   |  j  | } g  } |  j | <|  j } x� t d t | � | � D]� } | | | | !} d �  | D� } d j d �  | D� � } t j | � j } g  | D] \ } } }	 }
 |	 |
 f ^ q� } | j	 | | f � q@ Wd  S(   Ni    c         s�  s!   |  ] \ } } } } | Vq d  S(   N(    (   t   .0t   _R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    t   |c         s�  s   |  ] } d  | Vq d S(   s   (^%s$)N(    (   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    (
   R�   R�   t   _MAX_GROUPS_PER_PATTERNt   rangeR}   t   joinR�   R�   R�   R   (   RI   R�   t	   all_rulest
   comborulest	   maxgroupsR    t   somet   combinedR�   R�   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	+c   
      O�  s�   |  j  j | � } | s* t d | � � n  y� x( t | � D] \ } } | | d | <q: Wd j g  | D]- \ } } | r� | | j | � � n | ^ qe � }	 | s� |	 S|	 d t | � SWn+ t k
 r� t d t �  j	 d � � n Xd S(   s2    Build an URL by filling the wildcards in a rule. s   No route with that name.s   anon%dR�   t   ?s   Missing URL argument: %ri    N(
   R�   t   getR{   t	   enumerateR�   t   popR%   t   KeyErrorR   t   args(
   RI   t   _nameR�   t   queryR�   t   iRm   t   nt   ft   url(    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s      C c         C�  s<  | d j  �  } | d p d } d } | d k rG d | d d g } n d | d g } x� | D]� } | |  j k r� | |  j | k r� |  j | | \ } } | | r� | | � n i  f S| |  j k r] xc |  j | D]Q \ } }	 | | � }
 |
 r� |	 |
 j d \ } } | | r| | � n i  f Sq� Wq] q] Wt g  � } t | � } x> t |  j � | D]) } | |  j | k r]| j | � q]q]Wx_ t |  j � | | D]F } x= |  j | D]. \ } }	 | | � }
 |
 r�| j | � q�q�Wq�W| rd	 j t | � � } t	 d
 d d | �� n  t	 d d t
 | � � � d S(   sD    Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). t   REQUEST_METHODt	   PATH_INFOt   /t   HEADt   PROXYt   GETt   ANYi   t   ,i�  s   Method not allowed.t   Allowi�  s   Not found: N(   t   upperRg   R�   R�   t	   lastindexR\   R�   R�   t   sortedR�   t   repr(   RI   t   environt   verbR�   R�   t   methodsR�   R�   R�   R�   R�   t   allowedt   nocheckt   allow_header(    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s<    "'N(   RK   RL   Rp   R�   R�   R�   Rq   Rc   R�   R�   R�   R�   R�   Rg   R�   R�   R�   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �   s   
		F		t   Routec           B�  s�   e  Z d  Z d d d d � Z d �  Z e d �  � Z d �  Z d �  Z	 e
 d �  � Z d �  Z d �  Z d	 �  Z d
 �  Z d d � Z d �  Z RS(   s�    This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    c   	      K�  sp   | |  _  | |  _ | |  _ | |  _ | p- d  |  _ | p< g  |  _ | pK g  |  _ t �  j	 | d t
 �|  _ d  S(   Nt   make_namespaces(   t   appR�   R�   t   callbackRg   R�   t   pluginst   skiplistt
   ConfigDictt	   load_dictR�   t   config(	   RI   R�   R�   R�   R�   R�   R�   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    				c         O�  s   t  d � |  j | | �  S(   Ns�   Some APIs changed to return Route() instances instead of callables. Make sure to use the Route.call method and not to call Route instances directly.(   RY   t   call(   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    
c         C�  s
   |  j  �  S(   s�    The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.(   t   _make_callback(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   |  j  j d d � d S(   sk    Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. R�   N(   Rs   R�   Rg   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   reset�  s    c         C�  s   |  j  d S(   s:    Do all on-demand work immediately (useful for debugging).N(   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   prepare�  s    c         C�  sY   t  d � t d |  j d |  j d |  j d |  j d |  j d |  j d |  j d	 |  j	 � S(
   Ns=   Switch to Plugin API v2 and access the Route object directly.R�   R�   R�   R�   R�   R�   t   applyt   skip(
   RY   R]   R�   R�   R�   R�   R�   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _context�  s    
!c         c�  s�   t  �  } x� t |  j j |  j � D]� } t |  j k r< Pn  t | d t � } | ru | |  j k s# | | k ru q# n  | |  j k s# t | � |  j k r� q# n  | r� | j	 | � n  | Vq# Wd S(   s)    Yield all Plugins affecting this route. R�   N(
   R\   t   reversedR�   R�   R�   R�   Rh   Rq   t   typeR�   (   RI   t   uniqueR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   all_plugins�  s    	  ! $  c         C�  s�   |  j  } x� |  j �  D]� } ya t | d � rp t | d d � } | d k rR |  n |  j } | j | | � } n | | � } Wn t k
 r� |  j �  SX| |  j  k	 r t | |  j  � q q W| S(   NR�   t   apii   (	   R�   R   R2   Rh   R�   R�   Rx   R�   RN   (   RI   R�   t   pluginR  t   context(    (    s&   /home/lgardner/git/professor/bottle.pyR�   	  s    	c         C�  sx   |  j  } t | t r d n d | � } t r3 d n d } x8 t | | � rs t | | � rs t | | � d j } q< W| S(   sq    Return the callback. If the callback is a decorated function, try to
            recover the original function. t   __func__t   im_funct   __closure__t   func_closurei    (   R�   Rh   t   py3kR2   t   cell_contents(   RI   Rf   t   closure_attr(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_undecorated_callback  s    	!c         C�  s   t  |  j �  � d S(   s�    Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. i    (   R   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   get_callback_args#  s    c         C�  s8   x1 |  j  |  j j f D] } | | k r | | Sq W| S(   sp    Lookup a config field and return its value, first checking the
            route.config, then route.app.config.(   R�   R�   t   conifg(   RI   Ra   R
   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_config)  s     c         C�  s#   |  j  �  } d |  j |  j | f S(   Ns
   <%s %r %r>(   R  R�   R�   (   RI   t   cb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __repr__0  s    N(   RK   RL   Rp   Rg   Rc   R1   Rr   R�   R�   R�   t   propertyR�   R   R�   R  R  R  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s   						
	t   Bottlec           B�  s[  e  Z d  Z e e d � Z e d d � Z d& Z d Z e	 d �  � Z
 d �  Z d	 �  Z d
 �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d' d � Z d �  Z d �  Z d �  Z d �  Z d �  Z d' d d' d' d' d' d � Z d' d d � Z d' d d � Z d' d d � Z d' d d � Z d d  � Z d! �  Z  d" �  Z! d' d# � Z" d$ �  Z# d% �  Z$ RS((   s^   Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    c         C�  s�   t  �  |  _ t j |  j d � |  j _ |  j j d d t � |  j j d d t � | |  j d <| |  j d <t �  |  _	 g  |  _
 t �  |  _ i  |  _ g  |  _ |  j d r� |  j t �  � n  |  j t �  � d  S(   NR�   t   autojsont   validatet   catchall(   R�   R�   RM   t   partialt   trigger_hookt
   _on_changet   meta_sett   boolt   ResourceManagert	   resourcest   routesR�   t   routert   error_handlerR�   t   installt
   JSONPlugint   TemplatePlugin(   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   G  s    			R�   R  t   before_requestt   after_requestt	   app_resetc         C�  s   t  d �  |  j D� � S(   Nc         s�  s   |  ] } | g  f Vq d  S(   N(    (   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>f  s    (   R]   t   _Bottle__hook_names(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hooksd  s    c         C�  sA   | |  j  k r) |  j | j d | � n |  j | j | � d S(   s�   Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        i    N(   t   _Bottle__hook_reversedR'  t   insertR   (   RI   R�   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   add_hookh  s    c         C�  s>   | |  j  k r: | |  j  | k r: |  j  | j | � t Sd S(   s     Remove a callback from a hook. N(   R'  t   removeR�   (   RI   R�   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   remove_hookx  s    "c         O�  s(   g  |  j  | D] } | | | �  ^ q S(   s.    Trigger a hook and return a list of results. (   R'  (   RI   t   _Bottle__nameR�   t   kwargst   hook(    (    s&   /home/lgardner/git/professor/bottle.pyR  ~  s    c         �  s   �  � f d �  } | S(   se    Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details.c         �  s   � j  �  |  � |  S(   N(   R*  (   Rf   (   R�   RI   (    s&   /home/lgardner/git/professor/bottle.pyt	   decorator�  s    (    (   RI   R�   R0  (    (   R�   RI   s&   /home/lgardner/git/professor/bottle.pyR/  �  s    c         �  s  t  �  t � r t d t � n  g  | j d � D] } | r/ | ^ q/ } | s\ t d � � n  t | � � �  � f d �  } | j d t � | j d d � | j d i | d	 6�  d
 6� | | d <|  j d d j	 | � | � | j
 d � s|  j d d j	 | � | � n  d S(   s�   Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        s*   Parameter order of Bottle.mount() changed.R�   s   Empty path prefix.c          �  s�   z~ t  j � � t g  � �  d  �  f d � }  � t  j |  � } | rg �  j rg t j �  j | � } n  | ps �  j �  _ �  SWd  t  j � � Xd  S(   Nc         �  s[   | r! z t  | �  Wd  d  } Xn  |  �  _ x$ | D] \ } } �  j | | � q1 W�  j j S(   N(   R5   Rg   t   statust
   add_headert   bodyR   (   R1  t
   headerlistR   R�   Rm   (   t   rs(    s&   /home/lgardner/git/professor/bottle.pyt   start_response�  s    
	 (   t   requestt
   path_shiftt   HTTPResponseRg   R�   R3  t	   itertoolst   chain(   R6  R3  (   R�   t
   path_depth(   R5  s&   /home/lgardner/git/professor/bottle.pyt   mountpoint_wrapper�  s    	 R�   R�   R�   t
   mountpointR�   R�   R�   s   /%s/<:re:.*>N(   R>   t
   basestringRY   R�   t   splitR�   R}   R�   t   routeR�   t   endswith(   RI   R�   R�   t   optionsR�   t   segmentsR=  (    (   R�   R<  s&   /home/lgardner/git/professor/bottle.pyt   mount�  s    ( 
c         C�  s=   t  | t � r | j } n  x | D] } |  j | � q" Wd S(   s�    Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. N(   R>   R  R  t	   add_route(   RI   R  RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   merge�  s    c         C�  si   t  | d � r | j |  � n  t | � rK t  | d � rK t d � � n  |  j j | � |  j �  | S(   s�    Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        t   setupR�   s.   Plugins must be callable or implement .apply()(   R2   RH  t   callablet	   TypeErrorR�   R   R�   (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s     
c         C�  s�   g  | } } x� t  t |  j � � d d d � D]� \ } } | t k s~ | | k s~ | t | � k s~ t | d t � | k r0 | j | � |  j | =t | d � r� | j �  q� q0 q0 W| r� |  j	 �  n  | S(   s)   Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. Ni����R�   RJ   (
   R[   R�   R�   R�   R�   Rh   R   R2   RJ   R�   (   RI   R  t   removedR+  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   uninstall�  s    /*
  c         C�  s�   | d k r |  j } n+ t | t � r3 | g } n |  j | g } x | D] } | j �  qJ Wt r� x | D] } | j �  qk Wn  |  j d � d S(   s�    Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. R%  N(   Rg   R  R>   R�   R�   R�   R�   R  (   RI   RA  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s        c         C�  s=   x- |  j  D]" } t | d � r
 | j �  q
 q
 Wt |  _ d S(   s2    Close the application and all installed plugins. RJ   N(   R�   R2   RJ   R�   t   stopped(   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s     c         K�  s   t  |  | � d S(   s-    Calls :func:`run` with the same parameters. N(   t   run(   RI   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    c         C�  s   |  j  j | � S(   s�    Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match.(   R  R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         K�  sV   t  j j d d � j d � d } |  j j | | � j d � } t t d | � | � S(   s,    Return a string that matches a named route t   SCRIPT_NAMER�   R�   (   R7  R�   R�   t   stripR  R�   t   lstripR#   (   RI   t	   routenamet   kargst
   scriptnamet   location(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_url�  s    "c         C�  sL   |  j  j | � |  j j | j | j | d | j �t rH | j �  n  d S(   sS    Add a route object, but do not change the :data:`Route.app`
            attribute.R�   N(	   R  R   R  R�   R�   R�   R�   R�   R�   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyRF    s    % R�   c   	      �  si   t  � � r d � � } n  t | � � t | � � �  � � � � � � f d �  } | re | | � S| S(   s   A decorator to bind a function to a request URL. Example::

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
        c         �  s�   t  |  t � r t |  � }  n  xz t � � p6 t |  � D]` } xW t � � D]I } | j �  } t � | | |  d � d � d � �  �} � j | � qJ Wq7 W|  S(   NR�   R�   R�   (   R>   R?  t   loadR^   t   yieldroutesR�   R�   RF  (   R�   R�   R�   RA  (   R�   R�   R�   R�   R�   RI   R�   (    s&   /home/lgardner/git/professor/bottle.pyR0  &  s     N(   RI  Rg   R^   (	   RI   R�   R�   R�   R�   R�   R�   R�   R0  (    (   R�   R�   R�   R�   R�   RI   R�   s&   /home/lgardner/git/professor/bottle.pyRA    s     !
c         K�  s   |  j  | | | � S(   s    Equals :meth:`route`. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   2  s    t   POSTc         K�  s   |  j  | | | � S(   s8    Equals :meth:`route` with a ``POST`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   post6  s    t   PUTc         K�  s   |  j  | | | � S(   s7    Equals :meth:`route` with a ``PUT`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   put:  s    t   DELETEc         K�  s   |  j  | | | � S(   s:    Equals :meth:`route` with a ``DELETE`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete>  s    i�  c         �  s   �  � f d �  } | S(   s<    Decorator: Register an output handler for a HTTP error codec         �  s   |  � j  t �  � <|  S(   N(   R  R�   (   t   handler(   t   codeRI   (    s&   /home/lgardner/git/professor/bottle.pyRP   D  s    (    (   RI   R`  RP   (    (   R`  RI   s&   /home/lgardner/git/professor/bottle.pyR�   B  s    c         C�  s   t  t t d | �� S(   Nt   e(   RC   t   templatet   ERROR_PAGE_TEMPLATE(   RI   t   res(    (    s&   /home/lgardner/git/professor/bottle.pyt   default_error_handlerI  s    c         C�  s�  | d } | d <t  rY y  | j d � j d � | d <WqY t k
 rU t d d � SXn  y� |  | d <t j | � t j �  zT |  j d � |  j	 j
 | � \ } } | | d	 <| | d
 <| | d <| j | �  SWd  |  j d � XWn� t k
 r� t �  St k
 r| j �  |  j | � St t t f k
 r:�  nM t k
 r�|  j sV�  n  t �  } | d j | � t d d t �  | � SXd  S(   NR�   s   bottle.raw_pathR)   R=   i�  s#   Invalid path string. Expected UTF-8s
   bottle.appR#  s   route.handles   bottle.routes   route.url_argsR$  s   wsgi.errorsi�  s   Internal Server Error(   R  R@   RE   t   UnicodeErrorR�   R7  t   bindt   responseR  R  R�   R�   R9  R   Rx   R�   t   _handlet   KeyboardInterruptt
   SystemExitt   MemoryErrort	   ExceptionR  R   R   (   RI   R�   R�   RA  R�   t
   stacktrace(    (    s&   /home/lgardner/git/professor/bottle.pyRi  L  s>     





	 	c         C�  s$  | s# d t  k r d t  d <n  g  St | t t f � rn t | d t t f � rn | d d d !j | � } n  t | t � r� | j t  j � } n  t | t � r� d t  k r� t	 | � t  d <n  | g St | t
 � r| j t  � |  j j | j |  j � | � } |  j | � St | t � r=| j t  � |  j | j � St | d � r�d t j k rlt j d | � St | d � s�t | d � r�t | � Sn  y5 t | � } t | � } x | s�t | � } q�WWn� t k
 r�|  j d � St k
 rt �  } nW t t t f k
 r�  n; t k
 rY|  j s;�  n  t
 d d	 t �  t  �  � } n Xt | t � rv|  j | � St | t � r�t! j" | g | � } n_ t | t � r�d
 �  } t# | t! j" | g | � � } n& d t$ | � } |  j t
 d | � � St | d � r t% | | j& � } n  | S(   s�    Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        s   Content-Lengthi    t   reads   wsgi.file_wrapperRJ   t   __iter__R�   i�  s   Unhandled exceptionc         S�  s   |  j  t j � S(   N(   R@   Rh  t   charset(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   �  s    s   Unsupported response type: %s('   Rh  R>   RZ   R[   RA   R?   R�   R@   Rq  R}   R�   R�   R  R�   t   status_codeRe  t   _castR9  R3  R2   R7  R�   t   WSGIFileWrappert   iterR:   t   StopIterationR   Rj  Rk  Rl  Rm  R  R   R:  R;  R6   R�   t
   _closeiterRJ   (   RI   t   outt   peekt   ioutt   firstt   new_itert   encoderR�   (    (    s&   /home/lgardner/git/professor/bottle.pyRs  o  sh    !		 	!c         C�  sE  yw |  j  |  j | � � } t j d k s: | d d k r_ t | d � rV | j �  n  g  } n  | t j t j � | SWn� t t	 t
 f k
 r� �  n� t k
 r@|  j s� �  n  d t | j d	 d
 � � } t r| d t t t �  � � t t �  � f 7} n  | d j | � d g } | d | t j �  � t | � g SXd S(   s    The bottle WSGI-interface. id   ie   i�   i0  R�   R�   RJ   s4   <h1>Critical error while processing request: %s</h1>R�   R�   sD   <h2>Error:</h2>
<pre>
%s
</pre>
<h2>Traceback:</h2>
<pre>
%s
</pre>
s   wsgi.errorss   Content-Types   text/html; charset=UTF-8s   500 INTERNAL SERVER ERRORN(   id   ie   i�   i0  (   s   Content-Types   text/html; charset=UTF-8(   Rs  Ri  Rh  t   _status_codeR2   RJ   t   _status_lineR4  Rj  Rk  Rl  Rm  R  t   html_escapeR�   R�   R�   R   R   R   R   R   RC   (   RI   R�   R6  Rx  RF   t   headers(    (    s&   /home/lgardner/git/professor/bottle.pyt   wsgi�  s.     		 )	c         C�  s   |  j  | | � S(   s9    Each instance of :class:'Bottle' is a WSGI application. (   R�  (   RI   R�   R6  (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    (   s   before_requests   after_requests	   app_resets   configN(%   RK   RL   Rp   R�   Rc   R_   R  R&  R(  Rr   R'  R*  R,  R  R/  RE  RG  R   RL  Rg   R�   RJ   RN  R�   RV  RF  RA  R�   RZ  R\  R^  R�   Re  Ri  Rs  R�  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  >  s@   					0	
							)		#H	t   BaseRequestc           B�  s;  e  Z d  Z d Z d Z d@ d � Z e d d d e �d �  � Z	 e d d d e �d �  � Z
 e d d	 d e �d
 �  � Z e d �  � Z e d �  � Z e d d d e �d �  � Z d@ d � Z e d d d e �d �  � Z d@ d@ d � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z d �  Z d �  Z e d d d e �d  �  � Z d! �  Z e d" �  � Z e d# �  � Z e Z e d d$ d e �d% �  � Z e d& �  � Z  e d d' d e �d( �  � Z! e d) �  � Z" e d* �  � Z# e d+ �  � Z$ d, d- � Z% e d. �  � Z& e d/ �  � Z' e d0 �  � Z( e d1 �  � Z) e d2 �  � Z* e d3 �  � Z+ e d4 �  � Z, d5 �  Z- d@ d6 � Z. d7 �  Z/ d8 �  Z0 d9 �  Z1 d: �  Z2 d; �  Z3 d< �  Z4 d= �  Z5 d> �  Z6 d? �  Z7 RS(A   sd   A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    R�   i � c         C�  s,   | d k r i  n | |  _ |  |  j d <d S(   s!    Wrap a WSGI environ dictionary. s   bottle.requestN(   Rg   R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    s
   bottle.appRb   c         C�  s   t  d � � d S(   s+    Bottle application handling this request. s0   This request is not connected to an application.N(   t   RuntimeError(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    s   bottle.routec         C�  s   t  d � � d S(   s=    The bottle :class:`Route` object that matches this request. s)   This request is not connected to a route.N(   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRA  �  s    s   route.url_argsc         C�  s   t  d � � d S(   s'    The arguments extracted from the URL. s)   This request is not connected to a route.N(   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s    d |  j  j d d � j d � S(   s�    The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). R�   R�   R�   (   R�   R�   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    c         C�  s   |  j  j d d � j �  S(   s6    The ``REQUEST_METHOD`` value as an uppercase string. R�   R�   (   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    s   bottle.request.headersc         C�  s   t  |  j � S(   sf    A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. (   t   WSGIHeaderDictR�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j | | � S(   sA    Return the value of a request header, or a given default value. (   R�  R�   (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_header  s    s   bottle.request.cookiesc         C�  s5   t  |  j j d d � � j �  } t d �  | D� � S(   s�    Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. t   HTTP_COOKIER�   c         s�  s!   |  ] } | j  | j f Vq d  S(   N(   Ra   Rm   (   R�   t   c(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R*   R�   R�   t   valuest	   FormsDict(   RI   t   cookies(    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    !c         C�  sY   |  j  j | � } | rO | rO t | | � } | rK | d | k rK | d S| S| pX | S(   s   Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. i    i   (   R�  R�   t   cookie_decode(   RI   Ra   R
   t   secretRm   t   dec(    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_cookie  s
    "s   bottle.request.queryc         C�  sT   t  �  } |  j d <t |  j j d d � � } x | D] \ } } | | | <q6 W| S(   s    The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. s
   bottle.gett   QUERY_STRINGR�   (   R�  R�   t
   _parse_qslR�   (   RI   R�   t   pairsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   )  s
    s   bottle.request.formsc         C�  sI   t  �  } x9 |  j j �  D]( \ } } t | t � s | | | <q q W| S(   s   Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. (   R�  RY  t   allitemsR>   t
   FileUpload(   RI   t   formsR�   t   item(    (    s&   /home/lgardner/git/professor/bottle.pyR�  5  s
    	s   bottle.request.paramsc         C�  sa   t  �  } x' |  j j �  D] \ } } | | | <q Wx' |  j j �  D] \ } } | | | <qC W| S(   s�    A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. (   R�  R�   R�  R�  (   RI   t   paramsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  A  s    	s   bottle.request.filesc         C�  sI   t  �  } x9 |  j j �  D]( \ } } t | t � r | | | <q q W| S(   s�    File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        (   R�  RY  R�  R>   R�  (   RI   t   filesR�   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  L  s
    	s   bottle.request.jsonc         C�  sX   |  j  j d d � j �  j d � d } | d k rT |  j �  } | sJ d St | � Sd S(   s�    If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. t   CONTENT_TYPER�   t   ;i    s   application/jsonN(   R�   R�   t   lowerR@  t   _get_body_stringRg   t
   json_loads(   RI   t   ctypet   b(    (    s&   /home/lgardner/git/professor/bottle.pyt   jsonX  s    (
c         c�  sW   t  d |  j � } x> | rR | t | | � � } | s: Pn  | V| t | � 8} q Wd  S(   Ni    (   t   maxt   content_lengtht   minR}   (   RI   Ro  t   bufsizet   maxreadt   part(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _iter_bodyf  s    	 c         c�  s�  t  d d � } t d � t d � t d � } } } xYt r�| d � } xT | d | k r� | d � } | | 7} | s� | � n  t | � | k rM | � qM qM W| j | � \ }	 }
 }
 y t t |	 j �  � d � } Wn t k
 r� | � n X| d	 k rPn  | } xg | d	 k rq| s5| t	 | | � � } n  | |  | | } } | sY| � n  | V| t | � 8} qW| d
 � | k r8 | � q8 q8 Wd  S(   Ni�  s*   Error while parsing chunked transfer body.s   
R�  R�   i   i����i   i    i   (
   R�   RC   R�   R}   t	   partitionR�   t   tonatRP  R�   R�  (   RI   Ro  R�  RF   t   rnt   semt   bst   headerR�  t   sizeR�   R�  t   buffR�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _iter_chunkedn  s:    &	
 	 
  	s   bottle.request.bodyc         C�  s�   |  j  r |  j n |  j } |  j d j } t �  d t } } } x� | | |  j � D]n } | j | � | t	 | � 7} | rU | |  j k rU t
 d d � | } } | j | j �  � ~ t } qU qU W| |  j d <| j d � | S(   Ns
   wsgi.inputi    R�   s   w+b(   t   chunkedR�  R�  R�   Ro  R,   Rq   t   MEMFILE_MAXR   R}   R   t   getvalueR�   t   seek(   RI   t	   body_itert	   read_funcR3  t	   body_sizet   is_temp_fileR�  t   tmp(    (    s&   /home/lgardner/git/professor/bottle.pyt   _body�  s    c         C�  s�   |  j  } | |  j k r* t d d � � n  | d k  rF |  j d } n  |  j j | � } t | � |  j k r t d d � � n  | S(   s~    read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. i�  s   Request to largei    i   (   R�  R�  R�   R3  Ro  R}   (   RI   t   clenR   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	 c         C�  s   |  j  j d � |  j  S(   sl   The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. i    (   R�  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR3  �  s    c         C�  s   d |  j  j d d � j �  k S(   s(    True if Chunked transfer encoding was. R�  t   HTTP_TRANSFER_ENCODINGR�   (   R�   R�   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    s   bottle.request.postc   	      C�  sw  t  �  } |  j j d � s[ t t |  j �  d � � } x | D] \ } } | | | <q= W| Si d d 6} x1 d D]) } | |  j k ro |  j | | | <qo qo Wt d |  j d	 | d
 t	 � } t
 r� t | d d d d d �| d <n t r� d | d <n  t j | �  } | |  d <| j pg  } xR | D]J } | j r_t | j | j | j | j � | | j <q%| j | | j <q%W| S(   s�    The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        s
   multipart/R)   R�   R�  R�   R�  t   CONTENT_LENGTHt   fpR�   t   keep_blank_valuesR(   R=   t   newlines   
s   _cgi.FieldStorage(   s   REQUEST_METHODs   CONTENT_TYPER�  (   R�  t   content_typet
   startswithR�  R�  R�  R�   R]   R3  R�   t   py31RH   R  t   cgit   FieldStorageR[   t   filenameR�  t   fileR�   R�  Rm   (	   RI   RZ  R�  Ra   Rm   t   safe_envR�   R   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRY  �  s2    	 
	c         C�  s   |  j  j �  S(   s�    The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. (   t   urlpartst   geturl(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    s   bottle.request.urlpartsc         C�  s�   |  j  } | j d � p' | j d d � } | j d � pE | j d � } | s� | j d d � } | j d � } | r� | | d k r� d	 n d
 k r� | d | 7} q� n  t |  j � } t | | | | j d � d � S(   s�    The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. t   HTTP_X_FORWARDED_PROTOs   wsgi.url_schemet   httpt   HTTP_X_FORWARDED_HOSTt	   HTTP_HOSTt   SERVER_NAMEs	   127.0.0.1t   SERVER_PORTt   80t   443t   :R�  R�   (   R�   R�   t   urlquotet   fullpatht   UrlSplitResult(   RI   t   envR�  t   hostt   portR�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	!$c         C�  s   t  |  j |  j j d � � S(   s:    Request path including :attr:`script_name` (if present). R�   (   R#   t   script_nameR�   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  j d d � S(   sh    The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. R�  R�   (   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   query_string�  s    c         C�  s4   |  j  j d d � j d � } | r0 d | d Sd S(   s�    The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. RO  R�   R�   (   R�   R�   RP  (   RI   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    i   c         C�  s<   |  j  j d d � } t | |  j | � \ |  d <|  d <d S(   s�    Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        RO  R�   R�   N(   R�   R�   R8  R�   (   RI   t   shiftt   script(    (    s&   /home/lgardner/git/professor/bottle.pyR8  	  s    c         C�  s   t  |  j j d � p d � S(   s�    The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. R�  i����(   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j d d � j �  S(   sA    The Content-Type header as a lowercase-string (default: empty). R�  R�   (   R�   R�   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s%   |  j  j d d � } | j �  d k S(   s�    True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). t   HTTP_X_REQUESTED_WITHR�   t   xmlhttprequest(   R�   R�   R�  (   RI   t   requested_with(    (    s&   /home/lgardner/git/professor/bottle.pyt   is_xhr  s    c         C�  s   |  j  S(   s9    Alias for :attr:`is_xhr`. "Ajax" is not the right term. (   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   is_ajax'  s    c         C�  sK   t  |  j j d d � � } | r% | S|  j j d � } | rG | d f Sd S(   s�   HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. t   HTTP_AUTHORIZATIONR�   t   REMOTE_USERN(   t
   parse_authR�   R�   Rg   (   RI   t   basict   ruser(    (    s&   /home/lgardner/git/professor/bottle.pyt   auth,  s      
c         C�  sa   |  j  j d � } | r> g  | j d � D] } | j �  ^ q( S|  j  j d � } | r] | g Sg  S(   s(   A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. t   HTTP_X_FORWARDED_FORR�   t   REMOTE_ADDR(   R�   R�   R@  RP  (   RI   t   proxyt   ipt   remote(    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_route:  s
     &c         C�  s   |  j  } | r | d Sd S(   sg    The client IP as a string. Note that this information can be forged
            by malicious clients. i    N(   R�  Rg   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_addrE  s    	c         C�  s   t  |  j j �  � S(   sD    Return a new :class:`Request` with a shallow :attr:`environ` copy. (   t   RequestR�   t   copy(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  L  s    c         C�  s   |  j  j | | � S(   N(   R�   R�   (   RI   Rm   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   P  s    c         C�  s   |  j  | S(   N(   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __getitem__Q  s    c         C�  s   d |  | <|  j  | =d  S(   NR�   (   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delitem__R  s   
 c         C�  s   t  |  j � S(   N(   Ru  R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  S  s    c         C�  s   t  |  j � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __len__T  s    c         C�  s   |  j  j �  S(   N(   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   U  s    c         C�  s�   |  j  j d � r! t d � � n  | |  j  | <d } | d k rI d } n- | d
 k r^ d } n | j d � rv d } n  x% | D] } |  j  j d | d � q} Wd S(   sA    Change an environ value and clear all caches that depend on it. s   bottle.request.readonlys$   The environ dictionary is read-only.s
   wsgi.inputR3  R�  R�  R�  RZ  R�  R�  R�   t   HTTP_R�  R�  s   bottle.request.N(    (   s   bodys   formss   filess   paramss   posts   json(   s   querys   params(   s   headerss   cookies(   R�   R�   R�   R�  R�   Rg   (   RI   Ra   Rm   t   todelete(    (    s&   /home/lgardner/git/professor/bottle.pyt   __setitem__V  s    			c         C�  s   d |  j  j |  j |  j f S(   Ns   <%s: %s %s>(   t	   __class__RK   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  i  s    c         C�  s]   y5 |  j  d | } t | d � r0 | j |  � S| SWn! t k
 rX t d | � � n Xd S(   s@    Search in self.environ for additional user defined attributes. s   bottle.request.ext.%sRl   s   Attribute %r not defined.N(   R�   R2   Rl   R�   RO   (   RI   R�   t   var(    (    s&   /home/lgardner/git/professor/bottle.pyt   __getattr__l  s
    $c         C�  s4   | d k r t  j |  | | � S| |  j d | <d  S(   NR�   s   bottle.request.ext.%s(   t   objectt   __setattr__R�   (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  t  s     N(8   RK   RL   Rp   t	   __slots__R�  Rg   Rc   R_   R�   R�   RA  R�   R  R�   R�   R�  R�  R�  R�  R�   R�  R�  R�  R�  R�  R�  R�  R�  R3  R�  R�   RY  R�   R�  R�  R�  R�  R8  R�  R�  R�  R�  R�  R�  R�  R�  R�   R�  R�  Rp  R�  R�   R�  R  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  sd   			
#	
									c         C�  s   |  j  �  j d d � S(   NR�   t   -(   t   titlet   replace(   R0   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hkey{  s    t   HeaderPropertyc           B�  s5   e  Z d e d  d � Z d �  Z d �  Z d �  Z RS(   R�   c         C�  s=   | | |  _  |  _ | | |  _ |  _ d | j �  |  _ d  S(   Ns   Current value of the %r header.(   R�   R
   t   readert   writerR�  Rp   (   RI   R�   R  R  R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  sE   | d  k r |  S| j j |  j |  j � } |  j rA |  j | � S| S(   N(   Rg   R�  R�   R�   R
   R  (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �  s     c         C�  s   |  j  | � | j |  j <d  S(   N(   R  R�  R�   (   RI   Ri   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRn   �  s    c         C�  s   | j  |  j =d  S(   N(   R�  R�   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyRo   �  s    N(   RK   RL   Rg   R�   Rc   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR    s   		t   BaseResponsec        
   B�  s�  e  Z d  Z d Z d Z i e d+ � d 6e d, � d 6Z d d- d- d � Z d- d � Z	 d �  Z
 d �  Z e d �  � Z e d �  � Z d �  Z d �  Z e e e d- d � Z [ [ e d �  � Z d �  Z d �  Z d �  Z d �  Z d- d � Z d �  Z d �  Z d �  Z e d  �  � Z e d � Z e d d! e �Z e d" d! d# �  d$ d% �  �Z  e d& d' � � Z! d- d( � Z" d) �  Z# d* �  Z$ RS(.   s�   Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    i�   s   text/html; charset=UTF-8s   Content-Typei�   R�   s   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Md5s   Last-Modifiedi0  R�   c         K�  s�   d  |  _ i  |  _ | |  _ | p' |  j |  _ | r{ t | t � rQ | j �  } n  x' | D] \ } } |  j	 | | � qX Wn  | r� x- | j �  D] \ } } |  j	 | | � q� Wn  d  S(   N(
   Rg   t   _cookiest   _headersR3  t   default_statusR1  R>   R]   t   itemsR2  (   RI   R3  R1  R�  t   more_headersR�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    			c         C�  s�   | p	 t  } t | t  � s! t � | �  } |  j | _ t d �  |  j j �  D� � | _ |  j r� t �  | _ | j j	 |  j j
 d d � � n  | S(   s    Returns a copy of self. c         s�  s"   |  ] \ } } | | f Vq d  S(   N(    (   R�   t   kt   v(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    R�  R�   (   R  t
   issubclasst   AssertionErrorR1  R]   R  R	  R  R*   RW  t   output(   RI   Rj   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	"	"c         C�  s   t  |  j � S(   N(   Ru  R3  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    c         C�  s&   t  |  j d � r" |  j j �  n  d  S(   NRJ   (   R2   R3  RJ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s    c         C�  s   |  j  S(   s;    The HTTP status line as a string (e.g. ``404 Not Found``).(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   status_line�  s    c         C�  s   |  j  S(   s/    The HTTP status code as an integer (e.g. 404).(   R~  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRr  �  s    c         C�  s�   t  | t � r( | t j | � } } n= d | k rY | j �  } t | j �  d � } n t d � � d | k o| d k n s� t d � � n  | |  _ t | p� d | � |  _	 d  S(   Nt    i    s+   String status line without a reason phrase.id   i�  s   Status code out of range.s
   %d Unknown(
   R>   R�   t   _HTTP_STATUS_LINESR�   RP  R@  R�   R~  R�   R  (   RI   R1  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _set_status�  s     	c         C�  s   |  j  S(   N(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _get_status�  s    sQ   A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. c         C�  s   t  �  } |  j | _ | S(   sl    An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. (   t
   HeaderDictR  R]   (   RI   t   hdict(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	c         C�  s   t  | � |  j k S(   N(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __contains__�  s    c         C�  s   |  j  t | � =d  S(   N(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  t | � d S(   Ni����(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R  R  (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    |  j  j t | � | g � d S(   s|    Return the value of a previously defined header. If there is no
            header with that name, return a default value. i����(   R  R�   R  (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d S(   sh    Create a new response header, replacing any previously defined
            headers with the same name. N(   R�   R  R  (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_header   s    c         C�  s,   |  j  j t | � g  � j t | � � d S(   s=    Add an additional response header, not removing duplicates. N(   R  R�   R  R   R�   (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR2    s    c         C�  s   |  j  S(   sx    Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. (   R4  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iter_headers	  s    c   	      C�  s  g  } t  |  j j �  � } d |  j k rF | j d |  j g f � n  |  j |  j k r� |  j |  j } g  | D] } | d | k ro | ^ qo } n  | g  | D]% \ } } | D] } | | f ^ q� q� 7} |  j r	x3 |  j j �  D] } | j d | j	 �  f � q� Wn  | S(   s.    WSGI conform list of (header, value) tuples. s   Content-Typei    s
   Set-Cookie(
   R[   R  R	  R   t   default_content_typeR~  t   bad_headersR  R�  t   OutputString(	   RI   Rx  R�  R  t   hR�   t   valst   valR�  (    (    s&   /home/lgardner/git/professor/bottle.pyR4    s    ,6	 R  t   Expiresc         C�  s   t  j t |  � � S(   N(   R   t   utcfromtimestampt
   parse_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   !  s    R  c         C�  s
   t  |  � S(   N(   t	   http_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   "  s    s   UTF-8c         C�  s:   d |  j  k r6 |  j  j d � d j d � d j �  S| S(   sJ    Return the charset specified in the content-type header (default: utf8). s   charset=i����R�  i    (   R�  R@  RP  (   RI   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRq  $  s    'c         K�  sk  |  j  s t �  |  _  n  | r< t t | | f | � � } n t | t � sZ t d � � n  t | � d k r{ t d � � n  | |  j  | <x� | j	 �  D]� \ } } | d k r� t | t
 � r� | j | j d d } q� n  | d k rFt | t t f � r
| j �  } n' t | t t f � r1t j | � } n  t j d | � } n  | |  j  | | j d	 d
 � <q� Wd S(   s�   Create a new cookie or replace an old one. If the `secret` parameter is
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
        s)   Secret key missing for non-string Cookie.i   s   Cookie value to long.t   max_agei   i  t   expiress   %a, %d %b %Y %H:%M:%S GMTR�   R�  N(   R  R*   R/   t   cookie_encodeR>   R?  RJ  R}   R�   R	  R   t   secondst   dayst   datedateR   t	   timetupleR�   R�   t   timet   gmtimet   strftimeR   (   RI   R�   Rm   R�  RC  Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_cookie+  s(    !	 c         K�  s+   d | d <d | d <|  j  | d | � d S(   sq    Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. i����R$  i    R%  R�   N(   R.  (   RI   Ra   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete_cookiec  s    

c         C�  sD   d } x7 |  j  D], \ } } | d | j �  | j �  f 7} q W| S(   NR�   s   %s: %s
(   R4  R�  RP  (   RI   Rx  R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  j  s    $(   s   Content-Type(   s   Allows   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Types   Content-Md5s   Last-ModifiedN(%   RK   RL   Rp   R  R  R\   R  Rg   Rc   R�  Rp  RJ   R  R  Rr  R  R  R1  R�  R  R�  R�  R�  R�  R  R2  R  R4  R  R�  R�   R�  R%  Rq  R.  R/  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  sN    														8	c         �  s_   |  r t  d � n  t j �  �  �  f d �  } �  f d �  } �  f d �  } t | | | d � S(   Ns3   local_property() is deprecated and will be removed.c         �  s/   y �  j  SWn t k
 r* t d � � n Xd  S(   Ns    Request context not initialized.(   R�  RO   R�  (   RI   (   t   ls(    s&   /home/lgardner/git/professor/bottle.pyt   fgett  s     c         �  s   | �  _  d  S(   N(   R�  (   RI   Rm   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fsetx  s    c         �  s
   �  `  d  S(   N(   R�  (   RI   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fdely  s    s   Thread-local property(   RY   t	   threadingt   localR  (   R�   R1  R2  R3  (    (   R0  s&   /home/lgardner/git/professor/bottle.pyt   local_propertyq  s     t   LocalRequestc           B�  s    e  Z d  Z e j Z e �  Z RS(   sT   A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). (   RK   RL   Rp   R�  Rc   Rg  R6  R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR7  }  s   	t   LocalResponsec           B�  sD   e  Z d  Z e j Z e �  Z e �  Z e �  Z	 e �  Z
 e �  Z RS(   s+   A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    (   RK   RL   Rp   R  Rc   Rg  R6  R  R~  R  R  R3  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  �  s   					R9  c           B�  s#   e  Z d  d d d � Z d �  Z RS(   R�   c         K�  s#   t  t |  � j | | | | � d  S(   N(   t   superR9  Rc   (   RI   R3  R1  R�  R
  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  s@   |  j  | _  |  j | _ |  j | _ |  j | _ |  j | _ d  S(   N(   R~  R  R  R  R3  (   RI   Rh  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s
    N(   RK   RL   Rg   Rc   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR9  �  s   R�   c           B�  s#   e  Z d  Z d d d d d � Z RS(   i�  c         K�  s2   | |  _  | |  _ t t |  � j | | | � d  S(   N(   t	   exceptiont	   tracebackR9  R�   Rc   (   RI   R1  R3  R:  R;  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    		N(   RK   RL   R  Rg   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s   t   PluginErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR<  �  s    R!  c           B�  s)   e  Z d  Z d Z e d � Z d �  Z RS(   R�  i   c         C�  s   | |  _  d  S(   N(   R   (   RI   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         �  s)   |  j  � � s �  S�  � f d �  } | S(   Nc          �  s�   y �  |  | �  } Wn t  k
 r/ t �  } n Xt | t � rX � | � } d t _ | St | t � r� t | j t � r� � | j � | _ d | _ n  | S(   Ns   application/json(   R�   R   R>   R]   Rh  R�  R9  R3  (   R4   RR   t   rvt   json_response(   R�   R   (    s&   /home/lgardner/git/professor/bottle.pyRP   �  s    	!(   R   (   RI   R�   RA  RP   (    (   R�   R   s&   /home/lgardner/git/professor/bottle.pyR�   �  s
    	 (   RK   RL   R�   R  R   Rc   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR!  �  s   R"  c           B�  s#   e  Z d  Z d Z d Z d �  Z RS(   s   This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. Rb  i   c         C�  s{   | j  j d � } t | t t f � rT t | � d k rT t | d | d � | � St | t � rs t | � | � S| Sd  S(   NRb  i   i    i   (   R�   R�   R>   RZ   R[   R}   t   viewR�   (   RI   R�   RA  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    '(   RK   RL   Rp   R�   R  R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR"  �  s   t   _ImportRedirectc           B�  s&   e  Z d  �  Z d d � Z d �  Z RS(   c         C�  sv   | |  _  | |  _ t j j | t j | � � |  _ |  j j j	 i t
 d 6g  d 6g  d 6|  d 6� t j j |  � d S(   s@    Create a virtual package that redirects imports (see PEP 302). t   __file__t   __path__t   __all__t
   __loader__N(   R�   t   impmaskR   t   modulesR�   t   impt
   new_modulet   moduleRs   t   updateRA  t	   meta_pathR   (   RI   R�   RE  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    		!c         C�  s=   d | k r d  S| j  d d � d } | |  j k r9 d  S|  S(   Nt   .i   i    (   t   rsplitR�   (   RI   t   fullnameR�   t   packname(    (    s&   /home/lgardner/git/professor/bottle.pyt   find_module�  s      c         C�  s   | t  j k r t  j | S| j d d � d } |  j | } t | � t  j | } t  j | <t |  j | | � |  | _ | S(   NRL  i   (   R   RF  RM  RE  t
   __import__Ru   RI  RD  (   RI   RN  t   modnamet   realnameRI  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_module�  s     
	N(   RK   RL   Rc   Rg   RP  RT  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR@  �  s   		t	   MultiDictc           B�  s
  e  Z d  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 e r� d	 �  Z d
 �  Z d �  Z e
 Z e Z e Z e Z n? d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d d d d � Z d �  Z d �  Z d �  Z e Z e Z RS(   s�    This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    c         O�  s,   t  d �  t  | | �  j �  D� � |  _  d  S(   Nc         s�  s$   |  ] \ } } | | g f Vq d  S(   N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R	  (   RI   R4   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s    c         C�  s   t  |  j � S(   N(   R}   R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   t  |  j � S(   N(   Ru  R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp    s    c         C�  s   | |  j  k S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s    c         C�  s   |  j  | =d  S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  | d S(   Ni����(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  | | � d  S(   N(   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j �  S(   N(   R]   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s   |  ] } | d  Vq d S(   i����N(    (   R�   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s%   |  ] \ } } | | d  f Vq d S(   i����N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>   s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR	     s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R�   R  t   vlR  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>"  s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  !  s    c         C�  s$   g  |  j  j �  D] } | d ^ q S(   Ni����(   R]   R�  (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  )  s    c         C�  s0   g  |  j  j �  D] \ } } | | d f ^ q S(   Ni����(   R]   R	  (   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  *  s    c         C�  s   |  j  j �  S(   N(   R]   t   iterkeys(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRW  +  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s   |  ] } | d  Vq d S(   i����N(    (   R�   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>,  s    (   R]   t
   itervalues(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRX  ,  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s%   |  ] \ } } | | d  f Vq d S(   i����N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>.  s    (   R]   t	   iteritems(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRY  -  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R�   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>0  s    (   R]   RY  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iterallitems/  s    c         C�  s9   g  |  j  j �  D]% \ } } | D] } | | f ^ q  q S(   N(   R]   RY  (   RI   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  1  s    i����c         C�  sA   y) |  j  | | } | r$ | | � S| SWn t k
 r< n X| S(   s�   Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        (   R]   Rm  (   RI   Ra   R
   t   indexR�   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   4  s    
c         C�  s    |  j  j | g  � j | � d S(   s5    Add a new value to the list of values for this key. N(   R]   R�   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   E  s    c         C�  s   | g |  j  | <d S(   s1    Replace the list of values with a single value. N(   R]   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   I  s    c         C�  s   |  j  j | � p g  S(   s5    Return a (possibly empty) list of values for a key. (   R]   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   getallM  s    N(   RK   RL   Rp   Rc   R�  Rp  R  R�  R�  R�  R�   R  R�  R	  R�  RW  RX  RY  RZ  Rg   R�   R   R   R\  t   getonet   getlist(    (    (    s&   /home/lgardner/git/professor/bottle.pyRU    s<   																						R�  c           B�  sP   e  Z d  Z d Z e Z d d � Z d d � Z d d d � Z	 e
 �  d � Z RS(   s�   This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. R=   c         C�  sd   t  | t � r7 |  j r7 | j d � j | p3 |  j � St  | t � r\ | j | pX |  j � S| Sd  S(   NR)   (   R>   R?   t   recode_unicodeR@   RE   t   input_encodingRA   (   RI   R0   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _fixd  s
    c         C�  sq   t  �  } | p |  j } | _ t | _ xB |  j �  D]4 \ } } | j |  j | | � |  j | | � � q5 W| S(   s�    Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. (   R�  R`  Rq   R_  R�  R   Ra  (   RI   R(   R�  RB   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRE   l  s    		,c         C�  s7   y |  j  |  | | � SWn t t f k
 r2 | SXd S(   s7    Return the value as a unicode string, or the default. N(   Ra  Rf  R�   (   RI   R�   R
   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   getunicodew  s    c         C�  sG   | j  d � r4 | j d � r4 t t |  � j | � S|  j | d | �S(   Nt   __R
   (   R�  RB  R9  R�  R�  Rb  (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  ~  s    N(   RK   RL   Rp   R`  R�   R_  Rg   Ra  RE   Rb  R?   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  V  s   R  c           B�  sn   e  Z d  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 d d	 d
 � Z d �  Z RS(   sz    A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. c         O�  s,   i  |  _  | s | r( |  j | | �  n  d  S(   N(   R]   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	 c         C�  s   t  | � |  j k S(   N(   R  R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   |  j  t | � =d  S(   N(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  t | � d S(   Ni����(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s,   |  j  j t | � g  � j t | � � d  S(   N(   R]   R�   R  R   R�   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s    c         C�  s   |  j  j t | � � p g  S(   N(   R]   R�   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR\  �  s    i����c         C�  s   t  j |  t | � | | � S(   N(   RU  R�   R  (   RI   Ra   R
   R[  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  sJ   xC g  | D] } t  | � ^ q
 D]" } | |  j k r  |  j | =q  q  Wd  S(   N(   R  R]   (   RI   t   namesR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   filter�  s    &N(   RK   RL   Rp   Rc   R  R�  R�  R�  R   R   R\  Rg   R�   Re  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s   								R�  c           B�  sq   e  Z d  Z d Z d �  Z d �  Z d d � Z d �  Z d �  Z	 d �  Z
 d	 �  Z d
 �  Z d �  Z d �  Z RS(   s    This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    R�  R�  c         C�  s   | |  _  d  S(   N(   R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  s3   | j  d d � j �  } | |  j k r+ | Sd | S(   s6    Translate header field name to CGI/WSGI environ key. R�  R�   R�  (   R   R�   t   cgikeys(   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _ekey�  s    c         C�  s   |  j  j |  j | � | � S(   s:    Return the header value as is (may be bytes or unicode). (   R�   R�   Rg  (   RI   Ra   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt   raw�  s    c         C�  s   t  |  j |  j | � d � S(   NR)   (   R�  R�   Rg  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   t  d |  j � � d  S(   Ns   %s is read-only.(   RJ  R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   t  d |  j � � d  S(   Ns   %s is read-only.(   RJ  R�  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         c�  so   xh |  j  D]] } | d  d k r> | d j d d � j �  Vq
 | |  j k r
 | j d d � j �  Vq
 q
 Wd  S(   Ni   R�  R�   R�  (   R�   R   R�  Rf  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s
    c         C�  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   t  |  j �  � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  | � |  j k S(   N(   Rg  R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    (   s   CONTENT_TYPEs   CONTENT_LENGTHN(   RK   RL   Rp   Rf  Rc   Rg  Rg   Rh  R�  R�  R�  Rp  R�   R�  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   
								R�   c           B�  s�   e  Z d  Z d Z d e f d �  �  YZ d �  Z d �  Z d e d � Z	 d	 �  Z
 d
 �  Z d �  Z d �  Z d �  Z d d � Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z RS(   sH   A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    t   _metaR  t	   Namespacec           B�  s�   e  Z d  �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 d	 �  Z d
 �  Z d �  Z d �  Z d �  Z RS(   c         C�  s   | |  _  | |  _ d  S(   N(   t   _configt   _prefix(   RI   R�   t	   namespace(    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	c         C�  s    t  d � |  j |  j d | S(   Ns}   Accessing namespaces as dicts is discouraged. Only use flat item access: cfg["names"]["pace"]["key"] -> cfg["name.space.key"]RL  (   RY   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
c         C�  s   | |  j  |  j d | <d  S(   NRL  (   Rk  Rl  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  |  j d | =d  S(   NRL  (   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         c�  sZ   |  j  d } xF |  j D]; } | j d � \ } } } | |  j  k r | r | Vq q Wd  S(   NRL  (   Rl  Rk  t
   rpartition(   RI   t	   ns_prefixRa   t   nst   dotR�   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s
    c         C�  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   t  |  j �  � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  d | |  j k S(   NRL  (   Rl  Rk  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __str__�  s    c         C�  s�   t  d � | |  k rM | d j �  rM t j |  j |  j d | � |  | <n  | |  k rw | j d � rw t | � � n  |  j | � S(   Ns   Attribute access is deprecated.i    RL  Rc  (	   RY   t   isupperR�   Rj  Rk  Rl  R�  RO   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
'c         C�  s�   | d k r | |  j  | <d  St d � t t | � rE t d � � n  | |  k r� |  | r� t |  | |  j � r� t d � � n  | |  | <d  S(   NRk  Rl  s#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   s   _configs   _prefix(   Rs   RY   R2   R9   RO   R>   R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    
,c         C�  so   | |  k rk |  j  | � } t | |  j � rk | d } x. |  D]# } | j | � r> |  | | =q> q> Wqk n  d  S(   NRL  (   R�   R>   R�  R�  (   RI   Ra   R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delattr__  s    
c         O�  s   t  d � |  j | | �  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1     s    
(   RK   RL   Rc   R�  R�  R�  Rp  R�   R�  R  R  Rr  R�  R�  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRj  �  s   														c         O�  sB   i  |  _  d �  |  _ | s! | r> t d � |  j | | �  n  d  S(   Nc         S�  s   d  S(   N(   Rg   (   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    s-   Constructor does no longer accept parameters.(   Ri  R  RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s
    	
c         C�  sx   t  �  } | j | � x[ | j �  D]M } xD | j | � D]3 \ } } | d k rb | d | } n  | |  | <q9 Wq# W|  S(   s   Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        t   DEFAULTt   bottleRL  (   Ru  s   bottle(   R-   Ro  t   sectionsR	  (   RI   R�  R�   t   sectionRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_config!  s    	R�   c   	      C�  s  | | f g } x� | r| j  �  \ } } t | t � sR t d t | � � � n  x� | j �  D]� \ } } t | t � s� t d t | � � � n  | r� | d | n | } t | t � r� | j | | f � | r� |  j |  | � |  | <q� q_ | |  | <q_ Wq W|  S(   s�    Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        s   Source is not a dict (r)s   Key is not a string (%r)RL  (	   R�   R>   R]   RJ  R�   R	  R?  R   Rj  (	   RI   t   sourceRm  R�   t   stackR�   Ra   Rm   t   full_key(    (    s&   /home/lgardner/git/professor/bottle.pyR�   1  s    	c         O�  s{   d } | rC t  | d t � rC | d j d � d } | d } n  x1 t | | �  j �  D] \ } } | |  | | <qY Wd S(   s�    If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` R�   i    RL  i   N(   R>   R?  RP  R]   R	  (   RI   R4   RR   R�   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ  I  s    "c         C�  s!   | |  k r | |  | <n  |  | S(   N(    (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   T  s    c         C�  s�   t  | t � s( t d t | � � � n  |  j | d d �  � | � } | |  k rf |  | | k rf d  S|  j | | � t j |  | | � d  S(   Ns   Key has type %r (not a string)Re  c         S�  s   |  S(   N(    (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]  s    (   R>   R?  RJ  R�   t   meta_getR  R]   R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  Y  s    c         C�  s   t  j |  | � d  S(   N(   R]   R�  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  c  s    c         C�  s   x |  D] } |  | =q Wd  S(   N(    (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   clearf  s    c         C�  s   |  j  j | i  � j | | � S(   s-    Return the value of a meta field for a key. (   Ri  R�   (   RI   Ra   t	   metafieldR
   (    (    s&   /home/lgardner/git/professor/bottle.pyR}  j  s    c         C�  s:   | |  j  j | i  � | <| |  k r6 |  | |  | <n  d S(   sq    Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. N(   Ri  R�   (   RI   Ra   R  Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  n  s    c         C�  s   |  j  j | i  � j �  S(   s;    Return an iterable of meta field names defined for a key. (   Ri  R�   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   meta_listu  s    c         C�  sv   t  d � | |  k r? | d j �  r? |  j |  | � |  | <n  | |  k ri | j d � ri t | � � n  |  j | � S(   Ns   Attribute access is deprecated.i    Rc  (   RY   Rs  Rj  R�  RO   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  z  s    
c         C�  s�   | |  j  k r" t j |  | | � St d � t t | � rJ t d � � n  | |  k r� |  | r� t |  | |  j � r� t d � � n  | |  | <d  S(   Ns#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   R�  R]   R�  RY   R2   RO   R>   Rj  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
,c         C�  so   | |  k rk |  j  | � } t | |  j � rk | d } x. |  D]# } | j | � r> |  | | =q> q> Wqk n  d  S(   NRL  (   R�   R>   Rj  R�  (   RI   Ra   R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRt  �  s    
c         O�  s   t  d � |  j | | �  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    
(   s   _metas
   _on_changeN(   RK   RL   Rp   R�  R9   Rj  Rc   Ry  Rq   R�   RJ  R�   R�  R�  R~  Rg   R}  R  R�  R�  R�  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s$   A					
						
		t   AppStackc           B�  s#   e  Z d  Z d �  Z d d � Z RS(   s>    A stack-like list. Calling it returns the head of the stack. c         C�  s   |  d S(   s)    Return the current default application. i����(    (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    c         C�  s,   t  | t � s t �  } n  |  j | � | S(   s1    Add a new :class:`Bottle` instance to the stack (   R>   R  R   (   RI   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   push�  s    N(   RK   RL   Rp   R1   Rg   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	Rt  c           B�  s   e  Z d d � Z d �  Z RS(   i   i@   c         C�  sS   | | |  _  |  _ x9 d D]1 } t | | � r t |  | t | | � � q q Wd  S(   Nt   filenoRJ   Ro  t	   readlinest   tellR�  (   s   filenos   closes   reads	   readliness   tells   seek(   R�  t   buffer_sizeR2   Ru   Rh   (   RI   R�  R�  R`   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s     c         c�  s?   |  j  |  j } } x% t r: | | � } | s2 d  S| Vq Wd  S(   N(   R�  Ro  R�   (   RI   R�  Ro  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    	 i   (   RK   RL   Rc   Rp  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt  �  s   Rw  c           B�  s,   e  Z d  Z d d � Z d �  Z d �  Z RS(   s�    This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). c         C�  s   | |  _  t | � |  _ d  S(   N(   t   iteratorR^   t   close_callbacks(   RI   R�  RJ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	c         C�  s   t  |  j � S(   N(   Ru  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    c         C�  s   x |  j  D] } | �  q
 Wd  S(   N(   R�  (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s    N(   RK   RL   Rp   Rg   Rc   Rp  RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw  �  s   	R  c           B�  sP   e  Z d  Z d e d d � Z d	 d	 e d � Z d �  Z d �  Z	 d d � Z RS(
   sf   This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    s   ./t   allc         C�  s1   t  |  _ | |  _ | |  _ g  |  _ i  |  _ d  S(   N(   t   opent   openert   baset	   cachemodeR�   t   cache(   RI   R�  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s
    				c         C�  s�   t  j j t  j j | p |  j � � } t  j j t  j j | t  j j | � � � } | t  j 7} | |  j k r� |  j j | � n  | r� t  j j | � r� t  j	 | � n  | d k r� |  j j | � n |  j j | | � |  j j �  t  j j | � S(   s   Add a new path to the list of search paths. Return False if the
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
        N(   t   osR�   t   abspatht   dirnameR�  R�   t   sepR+  t   isdirt   makedirsRg   R   R)  R�  R~  t   exists(   RI   R�   R�  R[  t   create(    (    s&   /home/lgardner/git/professor/bottle.pyt   add_path�  s    '-c         c�  s�   |  j  } x� | r� | j �  } t j  j | � s7 q n  xS t j | � D]B } t j  j | | � } t j  j | � r� | j | � qG | VqG Wq Wd S(   s:    Iterate over all existing files in all registered paths. N(   R�   R�   R�  R�  t   listdirR�   R   (   RI   t   searchR�   R�   t   full(    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    
	  c         C�  s�   | |  j  k s t r� x[ |  j D]P } t j j | | � } t j j | � r |  j d k rk | |  j  | <n  | Sq W|  j d k r� d |  j  | <q� n  |  j  | S(   s�    Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. R�  t   found(   s   alls   foundN(   R�  R�   R�   R�  R�   t   isfileR�  Rg   (   RI   R�   R�   t   fpath(    (    s&   /home/lgardner/git/professor/bottle.pyt   lookup	  s    t   rc         O�  sA   |  j  | � } | s( t d | � � n  |  j | d | | | �S(   s=    Find a resource and return a file object, or raise IOError. s   Resource %r not found.R�   (   R�  t   IOErrorR�  (   RI   R�   R�   R�   R.  t   fname(    (    s&   /home/lgardner/git/professor/bottle.pyR�  	  s     N(
   RK   RL   Rp   R�  Rc   Rg   Rq   R�  Rp  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s   
		R�  c           B�  sb   e  Z d d  � Z e d � Z e d d e d d �Z e d �  � Z	 d d	 � Z
 e d d
 � Z RS(   c         C�  s=   | |  _  | |  _ | |  _ | r- t | � n t �  |  _ d S(   s    Wrapper for file uploads. N(   R�  R�   t   raw_filenameR  R�  (   RI   t   fileobjR�   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   "	  s    			s   Content-Types   Content-LengthR  R
   i����c         C�  s�   |  j  } t | t � s- | j d d � } n  t d | � j d d � j d � } t j j | j	 d t j j
 � � } t j d d | � j �  } t j d d	 | � j d
 � } | d  p� d S(   s�   Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        R=   t   ignoret   NFKDt   ASCIIs   \s   [^a-zA-Z0-9-_.\s]R�   s   [-\s]+R�  s   .-i�   t   empty(   R�  R>   R?   RE   R   R@   R�  R�   t   basenameR   R�  R�   R�   RP  (   RI   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  0	  s    
	$$i   i   c         C�  sa   |  j  j | j |  j  j �  } } } x$ | | � } | s? Pn  | | � q) W|  j  j | � d  S(   N(   R�  Ro  R   R�  R�  (   RI   R�  t
   chunk_sizeRo  R   R�   t   buf(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _copy_fileC	  s    & c         C�  s�   t  | t � r� t j j | � r< t j j | |  j � } n  | rd t j j | � rd t d � � n  t	 | d � � } |  j
 | | � Wd QXn |  j
 | | � d S(   s�   Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        s   File exists.t   wbN(   R>   R?  R�  R�   R�  R�   R�  R�  R�  R�  R�  (   RI   t   destinationt	   overwriteR�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   saveK	  s    	Ni   i   (   RK   RL   Rg   Rc   R  R�  R�   R�  Rr   R�  R�  Rq   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   	  s   i�  s   Unknown Error.c         C�  s   t  |  | � � d S(   s+    Aborts execution and causes a HTTP error. N(   R�   (   R`  t   text(    (    s&   /home/lgardner/git/professor/bottle.pyt   aborth	  s    c         C�  st   | s* t  j d � d k r! d n d } n  t j d t � } | | _ d | _ | j d t t  j	 |  � � | � d S(	   sd    Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. t   SERVER_PROTOCOLs   HTTP/1.1i/  i.  Rj   R�   t   LocationN(
   R7  R�   Rh  R�  R9  R1  R3  R  R#   R�   (   R�   R`  Rd  (    (    s&   /home/lgardner/git/professor/bottle.pyt   redirectm	  s    $		i   c         c�  s[   |  j  | � xG | d k rV |  j t | | � � } | s> Pn  | t | � 8} | Vq Wd S(   sF    Yield chunks from a range in a file. No chunk is bigger than maxread.i    N(   R�  Ro  R�  R}   (   R�  R�   RA   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _file_iter_rangey	  s     t   autos   UTF-8c         C�  s@  t  j j | � t  j } t  j j t  j j | |  j d � � � }  t �  } |  j | � sh t d d � St  j j	 |  � s� t  j j
 |  � r� t d d � St  j |  t  j � s� t d d � S| d k r� t j |  � \ } } | r� | | d <q� n  | r:| d	  d
 k r-| r-d | k r-| d | 7} n  | | d <n  | rut  j j | t k r[|  n | � } d | | d <n  t  j |  � } | j | d <} t j d t j | j � � }	 |	 | d <t j j d � }
 |
 r�t |
 j d � d j �  � }
 n  |
 d% k	 rD|
 t | j � k rDt j d t j �  � | d <t d d | � St j d k rYd n t  |  d � } d | d <t j j d � } d t j k r3t! t" t j d | � � } | s�t d d  � S| d \ } } d! | | d" | f | d# <t# | | � | d <| r t$ | | | | � } n  t | d d$ | �St | | � S(&   s�   Open a file in a safe way and return :exc:`HTTPResponse` with status
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
    s   /\i�  s   Access denied.i�  s   File does not exist.s/   You do not have permission to access this file.R�  s   Content-Encodingi   s   text/Rq  s   ; charset=%ss   Content-Types   attachment; filename="%s"s   Content-Dispositions   Content-Lengths   %a, %d %b %Y %H:%M:%S GMTs   Last-Modifiedt   HTTP_IF_MODIFIED_SINCER�  i    t   DateR1  i0  R�   R�   t   rbRA   s   Accept-Rangest
   HTTP_RANGEi�  s   Requested Range Not Satisfiables   bytes %d-%d/%di   s   Content-Rangei�   N(%   R�  R�   R�  R�  R�   RP  R]   R�  R�   R�  R�  t   accesst   R_OKt	   mimetypest
   guess_typeR�  R�   t   statt   st_sizeR+  R-  R,  t   st_mtimeR7  R�   R�   R"  R@  Rg   R�   R9  R�   R�  R[   t   parse_range_headerR�   R�  (   R�  t   roott   mimetypet   downloadRq  R�  R(   t   statsR�  t   lmt   imsR3  t   rangesR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   static_file�	  sX    *	& "$
"!$
 c         C�  s&   |  r t  j d � n  t |  � a d S(   sS    Change the debug level.
    There is only one debug level supported at the moment.R
   N(   RT   t   simplefilterR  R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   debug�	  s     c         C�  ss   t  |  t t f � r$ |  j �  }  n' t  |  t t f � rK t j |  � }  n  t  |  t � so t j	 d |  � }  n  |  S(   Ns   %a, %d %b %Y %H:%M:%S GMT(
   R>   R)  R   t   utctimetupleR�   R�   R+  R,  R?  R-  (   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR#  �	  s    c         C�  se   y@ t  j j |  � } t j | d  d � | d p6 d t j SWn t t t t	 f k
 r` d SXd S(   sD    Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. i   i    i	   N(   i    (   t   emailt   utilst   parsedate_tzR+  t   mktimet   timezoneRJ  R�   t
   IndexErrort   OverflowErrorRg   (   R�  t   ts(    (    s&   /home/lgardner/git/professor/bottle.pyR"  �	  s
    .c         C�  s�   ye |  j  d d � \ } } | j �  d k rd t t j t | � � � j  d d � \ } } | | f SWn t t f k
 r d SXd S(   s]    Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or Nonei   R�  R�  N(	   R@  Rg   R�  R/   t   base64t	   b64decodeRC   R�   R�   (   R�  R�   R   t   usert   pwd(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �	  s    -c         c�  s,  |  s |  d  d k r d Sg  |  d j  d � D]$ } d | k r/ | j  d d � ^ q/ } x� | D]� \ } } y� | s� t d | t | � � | } } nB | s� t | � | } } n& t | � t t | � d | � } } d | k o� | k  o� | k n r| | f Vn  Wq` t k
 r#q` Xq` Wd S(   s~    Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.i   s   bytes=NR�   R�  i   i    (   R@  R�  R�   R�  R�   (   R�  t   maxlenR�  R�  R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �	  s     >#&'c         C�  s�   g  } x� |  j  d d � j d � D]� } | s4 q" n  | j d d � } t | � d k rh | j d � n  t | d j  d d	 � � } t | d j  d d	 � � } | j | | f � q" W| S(
   NR�  t   &t   =i   i   R�   i    t   +R  (   R   R@  R}   R   t
   urlunquote(   t   qsR�  t   pairt   nvRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  
  s    "  c         C�  s6   t  d �  t |  | � D� � o5 t |  � t | � k S(   ss    Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. c         s�  s-   |  ]# \ } } | | k r! d  n d Vq d S(   i    i   N(    (   R�   R    t   y(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>
  s    (   t   sumt   zipR}   (   R4   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _lscmp
  s    c         C�  s^   t  j t j |  d � � } t  j t j t | � | � j �  � } t d � | t d � | S(   s>    Encode and sign a pickle-able object. Return a (byte) string i����t   !R�   (   R�  t	   b64encodet   pickleR   t   hmact   newRC   t   digest(   R   Ra   R�   t   sig(    (    s&   /home/lgardner/git/professor/bottle.pyR&  
  s    'c         C�  s�   t  |  � }  t |  � r� |  j t  d � d � \ } } t | d t j t j t  | � | � j �  � � r� t	 j
 t j | � � Sn  d S(   s?    Verify and decode an encoded string. Return an object or None.R�   i   N(   RC   t   cookie_is_encodedR@  R�  R�  R�  R�  R�  R�  R�  R   R�  Rg   (   R   Ra   R�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   
  s    4c         C�  s+   t  |  j t d � � o' t d � |  k � S(   s9    Return True if the argument looks like a encoded cookie.R�  R�   (   R  R�  RC   (   R   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  *
  s    c         C�  s@   |  j  d d � j  d d � j  d d � j  d d � j  d	 d
 � S(   s;    Escape HTML special characters ``&<>`` and quotes ``'"``. R�  s   &amp;t   <s   &lt;t   >s   &gt;t   "s   &quot;t   's   &#039;(   R   (   t   string(    (    s&   /home/lgardner/git/professor/bottle.pyR�  /
  s    *c         C�  s2   d t  |  � j d d � j d d � j d d � S(   s;    Escape and quote a string to be used as an HTTP attribute.s   "%s"s   
s   &#10;s   s   &#13;s   	s   &#9;(   R�  R   (   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt
   html_quote5
  s    c         c�  s�   d |  j  j d d � j d � } t |  � } t | d � t | d pK g  � } | d | t | d |  � 7} | Vx) | d | D] } | d | 7} | Vq� Wd S(   s�   Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    R�   Rc  i    i   s   /<%s>N(   RK   R   RQ  R   R}   RZ   (   Rf   R�   t   spect   argct   arg(    (    s&   /home/lgardner/git/professor/bottle.pyRX  ;
  s    
"$ c   	      C�  s}  | d k r |  | f S| j  d � j d � } |  j  d � j d � } | re | d d k re g  } n  | r� | d d k r� g  } n  | d k r� | t | � k r� | |  } | | } | | } nh | d k  r| t | � k r| | } | | } | |  } n( | d k  rd n d } t d | � � d d j | � } d d j | � } | j d � rs| rs| d 7} n  | | f S(   sS   Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    i    R�   R�   RO  R�   s"   Cannot shift. Nothing left from %s(   RP  R@  R}   R  R�   RB  (	   R�  t	   path_infoR�  t   pathlistt
   scriptlistt   movedR�  t   new_script_namet   new_path_info(    (    s&   /home/lgardner/git/professor/bottle.pyR8  O
  s.    	 
 	 	



 t   privates   Access deniedc         �  s   �  � � f d �  } | S(   se    Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. c         �  s   � �  � � f d �  } | S(   Nc          �  se   t  j p d \ } } | d  k s1 �  | | � rX t d � � } | j d d � � | S� |  | �  S(   Ni�  s   WWW-Authenticates   Basic realm="%s"(   NN(   R7  R�  Rg   R�   R2  (   R4   RR   R�  t   passwordRF   (   t   checkRf   t   realmR�  (    s&   /home/lgardner/git/professor/bottle.pyRP   r
  s    (    (   Rf   RP   (   R�  R   R�  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  q
  s    (    (   R�  R   R�  R0  (    (   R�  R   R�  s&   /home/lgardner/git/professor/bottle.pyt
   auth_basicn
  s    	c         �  s+   t  j t t �  � � �  f d �  � } | S(   sA    Return a callable that relays calls to the current default app. c          �  s   t  t �  �  � |  | �  S(   N(   Rh   R�   (   R4   RR   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyRP   �
  s    (   RM   t   wrapsRh   R  (   R�   RP   (    (   R�   s&   /home/lgardner/git/professor/bottle.pyt   make_default_app_wrapper�
  s    'RA  R�   RZ  R\  R^  R�   RE  R/  R   RL  RV  t   ServerAdapterc           B�  s/   e  Z e Z d  d d � Z d �  Z d �  Z RS(   s	   127.0.0.1i�  c         K�  s%   | |  _  | |  _ t | � |  _ d  S(   N(   RC  R�  R�   R�  (   RI   R�  R�  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �
  s    		c         C�  s   d  S(   N(    (   RI   R_  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    c         C�  sU   d j  g  |  j j �  D]" \ } } d | t | � f ^ q � } d |  j j | f S(   Ns   , s   %s=%ss   %s(%s)(   R�   RC  R	  R�   R�  RK   (   RI   R  R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s    A(   RK   RL   Rq   t   quietRc   RN  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   	t	   CGIServerc           B�  s   e  Z e Z d  �  Z RS(   c         �  s3   d d l  m } �  f d �  } | �  j | � d  S(   Ni����(   t
   CGIHandlerc         �  s   |  j  d d � �  |  | � S(   NR�   R�   (   R�   (   R�   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyt   fixed_environ�
  s    (   t   wsgiref.handlersR  RN  (   RI   R_  R  R  (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   R�   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   t   FlupFCGIServerc           B�  s   e  Z d  �  Z RS(   c         C�  sN   d d  l  } |  j j d |  j |  j f � | j j j | |  j � j �  d  S(   Ni����t   bindAddress(	   t   flup.server.fcgiRC  R�   R�  R�  t   servert   fcgit
   WSGIServerRN  (   RI   R_  t   flup(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR
  �
  s   t   WSGIRefServerc           B�  s   e  Z d  �  Z RS(   c         �  s�   d d l  m �  m } d d l  m } d d  l � d �  f �  � f d �  �  Y} � j j d | � } � j j d | � } d � j k r� t | d	 � � j	 k r� d
 | f � f d �  �  Y} q� n  | � j � j
 | | | � } | j �  d  S(   Ni����(   t   WSGIRequestHandlerR  (   t   make_servert   FixedHandlerc           �  s#   e  Z d  �  Z �  � f d �  Z RS(   c         S�  s   |  j  d S(   Ni    (   t   client_address(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   address_string�
  s    c          �  s   � j  s �  j |  | �  Sd  S(   N(   R  t   log_request(   R�   t   kw(   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s    	(   RK   RL   R  R  (    (   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   	t   handler_classt   server_classR�  t   address_familyt
   server_clsc           �  s   e  Z �  j Z RS(    (   RK   RL   t   AF_INET6R  (    (   t   socket(    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   (   t   wsgiref.simple_serverR  R  R  R  RC  R�   R�  Rh   t   AF_INETR�  t   serve_forever(   RI   R�   R  R  R  t   handler_clsR  t   srv(    (   R  RI   R  s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    "(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   t   CherryPyServerc           B�  s   e  Z d  �  Z RS(   c         C�  s�   d d l  m } |  j |  j f |  j d <| |  j d <|  j j d � } | r[ |  j d =n  |  j j d � } | r� |  j d =n  | j |  j �  } | r� | | _ n  | r� | | _ n  z | j	 �  Wd  | j
 �  Xd  S(   Ni����(   t
   wsgiservert	   bind_addrt   wsgi_appt   certfilet   keyfile(   t   cherrypyR%  R�  R�  RC  R�   t   CherryPyWSGIServert   ssl_certificatet   ssl_private_keyR�   t   stop(   RI   R_  R%  R(  R)  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s"    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR$  �
  s   t   WaitressServerc           B�  s   e  Z d  �  Z RS(   c         C�  s0   d d l  m } | | d |  j d |  j �d  S(   Ni����(   t   serveR�  R�  (   t   waitressR0  R�  R�  (   RI   R_  R0  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR/  �
  s   t   PasteServerc           B�  s   e  Z d  �  Z RS(   c         C�  se   d d l  m } d d l m } | | d |  j �} | j | d |  j d t |  j � |  j	 �d  S(   Ni����(   t
   httpserver(   t   TransLoggert   setup_console_handlerR�  R�  (
   t   pasteR3  t   paste.transloggerR4  R  R0  R�  R�   R�  RC  (   RI   R_  R3  R4  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s
    !(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR2  �
  s   t   MeinheldServerc           B�  s   e  Z d  �  Z RS(   c         C�  s:   d d l  m } | j |  j |  j f � | j | � d  S(   Ni����(   R  (   t   meinheldR  t   listenR�  R�  RN  (   RI   R_  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN     s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  �
  s   t   FapwsServerc           B�  s   e  Z d  Z d �  Z RS(   sA    Extremely fast webserver using libev. See http://www.fapws.org/ c         �  s�   d d  l  j } d d l m } m } |  j } t | j d � d k rV t | � } n  | j	 |  j
 | � d t j k r� |  j r� t d � t d � n  | j | � �  f d �  } | j d	 | f � | j �  d  S(
   Ni����(   R�  R�   i����g�������?t   BOTTLE_CHILDs3   WARNING: Auto-reloading does not work with Fapws3.
s/            (Fapws3 breaks python thread support)
c         �  s   t  |  d <�  |  | � S(   Ns   wsgi.multiprocess(   Rq   (   R�   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyR�     s    
R�   (   t   fapws._evwsgit   _evwsgit   fapwsR�  R�   R�  R�   t   SERVER_IDENTR�   R�   R�  R�  R�   R  t   _stderrt   set_base_modulet   wsgi_cbRN  (   RI   R_  t   evwsgiR�  R�   R�  R�   (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN    s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR;    s   t   TornadoServerc           B�  s   e  Z d  Z d �  Z RS(   s<    The super hyped asynchronous server by facebook. Untested. c         C�  s~   d d  l  } d d  l } d d  l } | j j | � } | j j | � } | j d |  j d |  j	 � | j
 j j �  j �  d  S(   Ni����R�  t   address(   t   tornado.wsgit   tornado.httpservert   tornado.ioloopR�  t   WSGIContainerR3  t
   HTTPServerR:  R�  R�  t   ioloopt   IOLoopt   instanceR�   (   RI   R_  t   tornadot	   containerR  (    (    s&   /home/lgardner/git/professor/bottle.pyRN    s
    $(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRE    s   t   AppEngineServerc           B�  s   e  Z d  Z e Z d �  Z RS(   s     Adapter for Google App Engine. c         �  sa   d d l  m � t j j d � } | rP t | d � rP �  � f d �  | _ n  � j �  � d  S(   Ni����(   t   utilR   t   mainc           �  s   � j  �  � S(   N(   t   run_wsgi_app(    (   R_  RR  (    s&   /home/lgardner/git/professor/bottle.pyR!   /  s    (   t   google.appengine.ext.webappRR  R   RF  R�   R2   RS  RT  (   RI   R_  RI  (    (   R_  RR  s&   /home/lgardner/git/professor/bottle.pyRN  )  s
    (   RK   RL   Rp   R�   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRQ  &  s   t   TwistedServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  s�   d d l  m } m } d d l m } d d l m } | �  } | j �  | j d d | j	 � | j
 | j | | | � � } | j |  j | d |  j �| j �  d  S(   Ni����(   R  R�  (   t
   ThreadPool(   t   reactort   aftert   shutdownt	   interface(   t   twisted.webR  R�  t   twisted.python.threadpoolRW  t   twisted.internetRX  R�   t   addSystemEventTriggerR.  t   Sitet   WSGIResourcet	   listenTCPR�  R�  RN  (   RI   R_  R  R�  RW  RX  t   thread_poolt   factory(    (    s&   /home/lgardner/git/professor/bottle.pyRN  5  s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRV  3  s   t   DieselServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  s3   d d l  m } | | d |  j �} | j �  d  S(   Ni����(   t   WSGIApplicationR�  (   t   diesel.protocols.wsgiRf  R�  RN  (   RI   R_  Rf  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRN  C  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRe  A  s   t   GeventServerc           B�  s   e  Z d  Z d �  Z RS(   s�    Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    c         �  s�   d d l  m } m } m } t t j �  | j � sI d } t | � � n  |  j j d d  � sg | } n  |  j
 rv d  n d |  j d <|  j |  j f } | j | | |  j � �  d t j k r� d d  l } | j | j �  f d �  � n  �  j �  d  S(	   Ni����(   R�  t   pywsgiR5  s9   Bottle requires gevent.monkey.patch_all() (before import)t   fastR
   t   logR<  c         �  s
   �  j  �  S(   N(   R.  (   R0   R�   (   R  (    s&   /home/lgardner/git/professor/bottle.pyR!   [  s    (   R   R�  Ri  R5  R>   R4  R�  RC  R�   Rg   R  R�  R�  R  R�  R�   t   signalt   SIGINTR!  (   RI   R_  R�  Ri  R5  R�   RF  Rl  (    (   R  s&   /home/lgardner/git/professor/bottle.pyRN  P  s     	(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRh  I  s   t   GeventSocketIOServerc           B�  s   e  Z d  �  Z RS(   c         C�  sB   d d l  m } |  j |  j f } | j | | |  j � j �  d  S(   Ni����(   R  (   t   socketioR  R�  R�  t   SocketIOServerRC  R!  (   RI   R_  R  RF  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  `  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRn  _  s   t   GunicornServerc           B�  s   e  Z d  Z d �  Z RS(   s?    Untested. See http://gunicorn.org/configure.html for options. c         �  ss   d d l  m } i d |  j t |  j � f d 6�  �  j |  j � d | f �  � f d �  �  Y} | �  j �  d  S(   Ni����(   t   Applications   %s:%dRg  t   GunicornApplicationc           �  s&   e  Z �  f d  �  Z � f d �  Z RS(   c         �  s   �  S(   N(    (   RI   t   parsert   optsR�   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyt   inito  s    c         �  s   �  S(   N(    (   RI   (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRW  r  s    (   RK   RL   Rv  RW  (    (   R�   R_  (    s&   /home/lgardner/git/professor/bottle.pyRs  n  s   (   t   gunicorn.app.baseRr  R�  R�   R�  RJ  RC  RN  (   RI   R_  Rr  Rs  (    (   R�   R_  s&   /home/lgardner/git/professor/bottle.pyRN  h  s
    #(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRq  f  s   t   EventletServerc           B�  s   e  Z d  Z d �  Z RS(   s
    Untested c         C�  s�   d d l  m } m } y0 | j | |  j |  j f � | d |  j �Wn3 t k
 r{ | j | |  j |  j f � | � n Xd  S(   Ni����(   R�  R:  t
   log_output(   t   eventletR�  R:  R  R�  R�  R  RJ  (   RI   R_  R�  R:  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  z  s    !(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx  x  s   t   RocketServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  sC   d d l  m } | |  j |  j f d i | d 6� } | j �  d  S(   Ni����(   t   RocketR�  R'  (   t   rocketR|  R�  R�  R�   (   RI   R_  R|  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    %(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{  �  s   t   BjoernServerc           B�  s   e  Z d  Z d �  Z RS(   s?    Fast server written in C: https://github.com/jonashaag/bjoern c         C�  s*   d d l  m } | | |  j |  j � d  S(   Ni����(   RN  (   t   bjoernRN  R�  R�  (   RI   R_  RN  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR~  �  s   t
   AutoServerc           B�  s,   e  Z d  Z e e e e e g Z d �  Z	 RS(   s    Untested. c         C�  sR   xK |  j  D]@ } y& | |  j |  j |  j � j | � SWq
 t k
 rI q
 Xq
 Wd  S(   N(   t   adaptersR�  R�  RC  RN  R   (   RI   R_  t   sa(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s
    &(
   RK   RL   Rp   R/  R2  RV  R$  R  R�  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   R�  R  R1  R*  R6  t   fapws3RO  t   gaet   twistedt   dieselR9  t   gunicornRz  t   geventSocketIOR}  R  c         K�  s�   d |  k r |  j  d d � n	 |  d f \ } }  | t j k rL t | � n  |  s] t j | S|  j �  r} t t j | |  � S| j  d � d } t j | | | <t d | |  f | � S(   s�   Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    R�  i   RL  i    s   %s.%sN(   R@  Rg   R   RF  RQ  t   isalnumRh   t   eval(   R�   Rm  RI  t   package_name(    (    s&   /home/lgardner/git/professor/bottle.pyRW  �  s    0   c         C�  sX   t  t a } z0 t j �  } t |  � } t | � r8 | S| SWd t j | � | a Xd S(   s�    Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. N(   R�   t   NORUNt   default_appR�  RW  RI  R+  (   R�   t   nr_oldR�  R=  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_app�  s    s	   127.0.0.1i�  c	         K�  s�  t  r
 d S| r~t j j d � r~z1yd }
 t j d d d d � \ } }
 t j | � x� t j j	 |
 � r=t
 j g t
 j } t j j �  } d | d <|
 | d <t j | d	 | �} x3 | j �  d k r� t j |
 d � t j | � q� W| j �  d
 k r] t j j	 |
 � r$t j |
 � n  t
 j | j �  � q] q] WWn t k
 rRn XWd t j j	 |
 � ryt j |
 � n  Xd Sy�| d k	 r�t | � n  |  p�t �  }  t |  t � r�t |  � }  n  t |  � s�t d |  � � n  x! | p�g  D] } |  j | � q�W| t k r(t j | � } n  t | t � rFt  | � } n  t | t! � rp| d | d | |	 � } n  t | t" � s�t d | � � n  | j# p�| | _# | j# s�t$ d t% t& | � f � t$ d | j' | j( f � t$ d � n  | rQt j j d � }
 t) |
 | � } | � | j* |  � Wd QX| j+ d k r^t
 j d
 � q^n | j* |  � Wnr t k
 rrnb t, t- f k
 r��  nI | s��  n  t. | d | � s�t/ �  n  t j | � t
 j d
 � n Xd S(   s�   Start a server instance. This method blocks until the server terminates.

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
     NR<  R�   s   bottle.t   suffixs   .lockt   truet   BOTTLE_LOCKFILER�  i   s   Application is not callable: %rR�  R�  s!   Unknown or unsupported server: %rs,   Bottle v%s server starting up (using %s)...
s   Listening on http://%s:%d/
s   Hit Ctrl-C to quit.

t   reloadR  (0   R�  R�  R�   R�   Rg   t   tempfilet   mkstempRJ   R�   R�  R   t
   executablet   argvR�  t
   subprocesst   Popent   pollt   utimeR+  t   sleept   unlinkt   exitRj  t   _debugR�  R>   R?  R�  RI  R�   R   t   server_namesRW  R�   R  R  RA  t   __version__R�   R�  R�  t   FileCheckerThreadRN  R1  Rk  Rl  Rh   R   (   R�   R  R�  R�  t   intervalt   reloaderR  R�   R�  RS  t   lockfilet   fdR�   R�   R�   R  t   bgcheck(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s�      

  	 
R�  c           B�  s2   e  Z d  Z d �  Z d �  Z d �  Z d �  Z RS(   sw    Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. c         C�  s0   t  j j |  � | | |  _ |  _ d  |  _ d  S(   N(   R4  t   ThreadRc   R�  R�  Rg   R1  (   RI   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ?  s    c         C�  s[  t  j j } d �  } t �  } xq t t j j �  � D]Z } t | d d � } | d d k ri | d  } n  | r4 | | � r4 | | � | | <q4 q4 Wx� |  j	 sV| |  j
 � s� | |  j
 � t j �  |  j d k  r� d	 |  _	 t j �  n  xV t | j �  � D]B \ } } | | � s(| | � | k r� d
 |  _	 t j �  Pq� q� Wt j |  j � q� Wd  S(   Nc         S�  s   t  j |  � j S(   N(   R�  R�  R�  (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!   G  s    RA  R�   i����s   .pyos   .pyci����i   R�   R�  (   s   .pyos   .pyc(   R�  R�   R�  R]   R[   R   RF  R�  Rh   R1  R�  R+  R�  t   threadt   interrupt_mainR	  R�  (   RI   R�  t   mtimeR�  RI  R�   t   lmtime(    (    s&   /home/lgardner/git/professor/bottle.pyRN  E  s(    		  &		
c         C�  s   |  j  �  d  S(   N(   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   __enter__[  s    c         C�  s8   |  j  s d |  _  n  |  j �  | d  k	 o7 t | t � S(   NR�  (   R1  R�   Rg   R  Rj  (   RI   t   exc_typet   exc_valt   exc_tb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __exit__^  s    	 
(   RK   RL   Rp   Rc   RN  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  ;  s
   			t   TemplateErrorc           B�  s   e  Z d  �  Z RS(   c         C�  s   t  j |  d | � d  S(   Ni�  (   R�   Rc   (   RI   RW   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   m  s    (   RK   RL   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  l  s   t   BaseTemplatec           B�  st   e  Z d  Z d d d d g Z i  Z i  Z d d g  d d � Z e g  d � � Z	 e d �  � Z
 d	 �  Z d
 �  Z RS(   s2    Base class and minimal API for template adapters t   tplt   htmlt   thtmlt   stplR=   c         K�  s+  | |  _  t | d � r$ | j �  n | |  _ t | d � rE | j n d |  _ g  | D] } t j j | � ^ qU |  _	 | |  _
 |  j j �  |  _ |  j j | � |  j r� |  j  r� |  j |  j  |  j	 � |  _ |  j s� t d t | � � � q� n  |  j r|  j rt d � � n  |  j |  j �  d S(   s=   Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        Ro  R�  s   Template %s not found.s   No template specified.N(   R�   R2   Ro  Rz  R�  Rg   R�  R�   R�  R�  R(   t   settingsR�  RJ  R�  R�  R�   R�   (   RI   Rz  R�   R�  R(   R�  R    (    (    s&   /home/lgardner/git/professor/bottle.pyRc   w  s    	$!(		c         C�  s  | s t  d � d g } n  t j j | � rZ t j j | � rZ t  d � t j j | � Sx� | D]� } t j j | � t j } t j j t j j | | � � } | j | � s� qa n  t j j | � r� | Sx; |  j	 D]0 } t j j d | | f � r� d | | f Sq� Wqa Wd S(   s{    Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. s2   The template lookup path list should not be empty.RL  s,   Absolute template path names are deprecated.s   %s.%sN(
   RY   R�  R�   t   isabsR�  R�  R�  R�   R�  t
   extensions(   Rj   R�   R�  t   spathR�  t   ext(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s     
$
!  c         G�  s;   | r, |  j  j �  |  _  | d |  j  | <n |  j  | Sd S(   sB    This reads or sets the global settings stored in class.settings. i    N(   R�  R�  (   Rj   Ra   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   global_config�  s    c         K�  s
   t  � d S(   s�    Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        N(   t   NotImplementedError(   RI   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         O�  s
   t  � d S(   sF   Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        N(   R�  (   RI   R�   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   render�  s    N(   RK   RL   Rp   R�  R�  t   defaultsRg   Rc   t   classmethodR�  R�  R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  q  s   		t   MakoTemplatec           B�  s   e  Z d  �  Z d �  Z RS(   c         K�  s�   d d l  m } d d l m } | j i |  j d 6� | j d t t � � | d |  j	 | � } |  j
 r� | |  j
 d | | �|  _ n' | d |  j d	 |  j d | | � |  _ d  S(
   Ni����(   t   Template(   t   TemplateLookupR`  t   format_exceptionst   directoriesR�  t   uriR�  (   t   mako.templateR�  t   mako.lookupR�  RJ  R(   R�   R  R�   R�  Rz  R�  R�   R�  (   RI   RC  R�  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	c         O�  sJ   x | D] } | j  | � q W|  j j �  } | j  | � |  j j | �  S(   N(   RJ  R�  R�  R�  R�  (   RI   R�   R.  t   dictargt	   _defaults(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     (   RK   RL   R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	t   CheetahTemplatec           B�  s   e  Z d  �  Z d �  Z RS(   c         K�  s~   d d l  m } t j �  |  _ i  |  j _ |  j j g | d <|  j rb | d |  j | � |  _ n | d |  j | � |  _ d  S(   Ni����(   R�  t
   searchListRz  R�  (	   t   Cheetah.TemplateR�  R4  R5  R  t   varsRz  R�  R�  (   RI   RC  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	c         O�  sj   x | D] } | j  | � q W|  j j j  |  j � |  j j j  | � t |  j � } |  j j j �  | S(   N(   RJ  R  R�  R�  R�   R�  R~  (   RI   R�   R.  R�  Rx  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s     (   RK   RL   R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	
t   Jinja2Templatec           B�  s,   e  Z d d i  d  � Z d �  Z d �  Z RS(   c         K�  s�   d d l  m } m } d | k r1 t d � � n  | d | |  j � | � |  _ | rk |  j j j | � n  | r� |  j j j | � n  | r� |  j j	 j | � n  |  j
 r� |  j j |  j
 � |  _ n |  j j |  j � |  _ d  S(   Ni����(   t   Environmentt   FunctionLoaderR�   ss   The keyword argument `prefix` has been removed. Use the full jinja2 environment name line_statement_prefix instead.t   loader(   t   jinja2R�  R�  R�  R�  R�  R�   RJ  t   testst   globalsRz  t   from_stringR�  t   get_templateR�  (   RI   R�   R�  R�  R.  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s       	c         O�  sJ   x | D] } | j  | � q W|  j j �  } | j  | � |  j j | �  S(   N(   RJ  R�  R�  R�  R�  (   RI   R�   R.  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     c         C�  sQ   |  j  | |  j � } | s d  St | d � � } | j �  j |  j � SWd  QXd  S(   NR�  (   R�  R�  R�  Ro  RE   R(   (   RI   R�   R�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     N(   RK   RL   Rg   R�   R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	t   SimpleTemplatec           B�  sb   e  Z e e d d  � Z e d �  � Z e d �  � Z d d � Z	 d d � Z
 d �  Z d �  Z RS(   c         �  sh   i  |  _  |  j �  �  f d �  |  _ �  � f d �  |  _ | |  _ | rd |  j |  j |  _ |  _ n  d  S(   Nc         �  s   t  |  �  � S(   N(   R/   (   R    (   RB   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    c         �  s   � t  |  �  � � S(   N(   R/   (   R    (   RB   t   escape_func(    s&   /home/lgardner/git/professor/bottle.pyR!   	  s    (   R�  R(   t   _strt   _escapet   syntax(   RI   R�  t   noescapeR�  RR   (    (   RB   R�  s&   /home/lgardner/git/professor/bottle.pyR�     s    			c         C�  s   t  |  j |  j p d d � S(   Ns   <string>R<   (   R�   R`  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   co  s    c         C�  s�   |  j  } | s9 t |  j d � � } | j �  } Wd  QXn  y t | � d } } Wn1 t k
 r� t d � t | d � d } } n Xt | d | d |  j �} | j	 �  } | j
 |  _
 | S(   NR�  R=   s;   Template encodings other than utf8 are no longer supported.R)   R(   R�  (   Rz  R�  R�  Ro  R/   Rf  RY   t
   StplParserR�  t	   translateR(   (   RI   Rz  R�   R(   Rt  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR`    s    	
c         K�  s0   | d  k r t d t � n  | | f | d <d  S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?t   _rebase(   Rg   RY   R�   (   RI   t   _envR�   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  "  s    
c         K�  s�   | d  k r t d t � n  | j �  } | j | � | |  j k ri |  j d | d |  j � |  j | <n  |  j | j | d | � S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?R�   R�  t   _stdout(	   Rg   RY   R�   R�  RJ  R�  R�  R�  t   execute(   RI   R�  R�   R.  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _include(  s    
%c         C�  s  |  j  j �  } | j | � | j i
 | d 6| j d 6t j |  j | � d 6t j |  j | � d 6d  d 6|  j	 d 6|  j
 d 6| j d 6| j d	 6| j d
 6� t |  j | � | j d � r� | j d � \ } } d j | � | d <| 2|  j | | | � S| S(   NR�  t
   _printlistt   includet   rebaseR�  R�  R�  R�   R�   t   definedR�   R�  (   R�  R�  RJ  t   extendRM   R  R�  R�  Rg   R�  R�  R�   R�   R  R�  R�  R�   R�   (   RI   R�  R.  R�  t   subtplt   rargs(    (    s&   /home/lgardner/git/professor/bottle.pyR�  2  s    c         O�  sT   i  } g  } x | D] } | j  | � q W| j  | � |  j | | � d j | � S(   sA    Render the template using keyword arguments as local variables. R�   (   RJ  R�  R�   (   RI   R�   R.  R�  R   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  B  s      N(   RK   RL   R�  Rq   Rg   R�   Rr   R�  R`  R�  R�  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s   	
	t   StplSyntaxErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  K  s    R�  c           B�  s�   e  Z d  Z i  Z d Z e j d d � Z e d 7Z e d 7Z e d 7Z e d 7Z e d 7Z e d	 7Z e d
 7Z d Z d e Z d Z d d d � Z
 d �  Z d �  Z e e e � Z d �  Z d �  Z d �  Z d �  Z d d � Z d �  Z RS(   s    Parser for stpl templates. s�   ((?m)[urbURB]?(?:''(?!')|""(?!")|'{6}|"{6}|'(?:[^\\']|\\.)+?'|"(?:[^\\"]|\\.)+?"|'{3}(?:[^\\]|\\.|\n)+?'{3}|"{3}(?:[^\\]|\\.|\n)+?"{3}))s   |\nR�   s   |(#.*)s   |([\[\{\(])s   |([\]\}\)])sW   |^([ \t]*(?:if|for|while|with|try|def|class)\b)|^([ \t]*(?:elif|else|except|finally)\b)s?   |((?:^|;)[ \t]*end[ \t]*(?=(?:%(block_close)s[ \t]*)?\r?$|;|#))s   |(%(block_close)s[ \t]*(?=$))s   |(\r?\n)s8   (?m)^[ 	]*(\\?)((%(line_start)s)|(%(block_start)s))(%%?)s2   %%(inline_start)s((?:%s|[^'"
]*?)+)%%(inline_end)ss   <% %> % {{ }}R=   c         C�  sv   t  | | � | |  _ |  _ |  j | p. |  j � g  g  |  _ |  _ d \ |  _ |  _ d \ |  _	 |  _
 d |  _ d  S(   Ni   i    (   i   i    (   i    i    (   R/   Rz  R(   t
   set_syntaxt   default_syntaxt   code_buffert   text_buffert   linenoR�   t   indentt
   indent_modt   paren_depth(   RI   Rz  R�  R(   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   n  s    c         C�  s   |  j  S(   s=    Tokens as a space separated string (default: <% %> % {{ }}) (   t   _syntax(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_syntaxv  s    c         C�  s�   | |  _  | j �  |  _ | |  j k r� d } t t j |  j � } t t | j �  | � � } |  j	 |  j
 |  j f } g  | D] } t j | | � ^ q| } | |  j | <n  |  j | \ |  _ |  _ |  _ d  S(   Ns:   block_start block_close line_start inline_start inline_end(   R�  R@  t   _tokenst	   _re_cachet   mapR�   R�   R]   R�  t	   _re_splitt   _re_tokt   _re_inlR�   t   re_splitt   re_tokt   re_inl(   RI   R�  Rd  t   etokenst   pattern_varst   patternsR�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  z  s    	&c         C�  s�  |  j  r t d � � n  x�t r�|  j j |  j |  j  � } | r�|  j |  j  |  j  | j �  !} |  j j | � |  j  | j	 �  7_  | j
 d � r
|  j |  j  j d � \ } } } |  j j | j
 d � | j
 d � | | � |  j  t | | � d 7_  q n | j
 d � r�t d � |  j |  j  j d � \ } } } |  j j | j
 d � | | � |  j  t | | � d 7_  q n  |  j �  |  j d t | j
 d � � � q Pq W|  j j |  j |  j  � |  j �  d	 j |  j � S(
   Ns   Parser is a one time instance.i   s   
i   i   s#   Escape code lines with a backslash.t	   multilinei   R�   (   R�   R�  R�   R�  R�  Rz  R�   R�  R   R�   R~   R�  R}   RY   t
   flush_textt	   read_codeR  R�   R�  (   RI   R   R�  t   lineR�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s2    	 	 ".
"!
"
c      	   C�  su  d \ } } xbt  rp|  j j |  j |  j � } | sw | |  j |  j 7} t |  j � |  _ |  j | j �  | � d  S| |  j |  j |  j | j �  !7} |  j | j	 �  7_ | j
 �  \	 } } } } }	 }
 } } } | s� |  j d k r|	 s� |
 r| |	 p|
 7} q n  | r!| | 7} q | r[| } | rm| j �  j |  j d � rmt } qmq | r}|  j d 7_ | | 7} q | r�|  j d k r�|  j d 8_ n  | | 7} q |	 r�|	 d } |  _ |  j d 7_ q |
 r�|
 d } |  _ q | r
|  j d 8_ q | r,| rt } qm| | 7} q |  j | j �  | � |  j d 7_ d \ } } |  _ | s Pq q Wd  S(   NR�   i    i   i����(   R�   R�   (   R�   R�   i    (   R�   R   R�  Rz  R�   R}   t
   write_codeRP  R�   R�   R�   R�  RB  R�  Rq   R�  R�  R�  (   RI   R  t	   code_linet   commentR   R�  t   _comt   _pot   _pct   _blk1t   _blk2t   _endt   _cendt   _nl(    (    s&   /home/lgardner/git/professor/bottle.pyR  �  sV    	$'!" 	c   	      C�  s�  d j  |  j � } |  j 2| s# d  Sg  d d d |  j } } } x� |  j j | � D]� } | | | j �  !| j �  } } | r� | j | j  t t	 | j
 t � � � � n  | j d � r� | d c | 7<n  | j |  j | j d � j �  � � qU W| t | � k  r�| | } | j
 t � } | d j d � rJ| d d	  | d <n( | d j d
 � rr| d d  | d <n  | j | j  t t	 | � � � n  d d j  | � } |  j | j d � d 7_ |  j | � d  S(   NR�   i    s   \
s     s   
i����i   s   \\
i����s   \\
i����s   _printlist((%s,))s   , (   R�   R�  R�  R  R�   R�   R�   R   R�  R�   t
   splitlinesR�   RB  t   process_inlineR~   RP  R}   R�  t   countR	  (	   RI   R�  t   partst   post   nlR   R�   t   linesR`  (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s.      + )
  "c         C�  s$   | d d k r d | d Sd | S(   Ni    R�  s   _str(%s)i   s   _escape(%s)(    (   RI   t   chunk(    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s     c         C�  sX   |  j  | | � \ } } d |  j |  j } | | j �  | d 7} |  j j | � d  S(   Ns     s   
(   t   fix_backward_compatibilityR�  R�  RQ  R�  R   (   RI   R  R  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  �  s    c         C�  s7  | j  �  j d  d � } | r� | d d k r� t d � t | � d k rT d | f St | � d k rz d t | � | f Sd	 t | � | f Sn  |  j d k r-| j  �  r-d
 | k r-t j d | � } | r-t d � | j	 d � } |  j
 j |  j � j | � |  _
 | |  _ | | j d
 d � f Sn  | | f S(   Ni   i    R�  R�  s2   The include and rebase keywords are functions now.i   s   _printlist([base])s   _=%s(%r)s   _=%s(%r, %s)t   codings   #.*coding[:=]\s*([-\w.]+)s4   PEP263 encoding strings in templates are deprecated.s   coding*(   s   includes   rebase(   RP  R@  Rg   RY   R}   RZ   R�  R�   R�   R~   Rz  R@   R(   RE   R   (   RI   R  R  R  R   RB   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s"    
 
 (
!	N(   RK   RL   Rp   R�  R�  R   R�  R�  R�  Rg   Rc   R�  R�  R  R�  R�  R  R  R  R	  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  N  s0   







				/		c          O�  se  |  r |  d n d } | j d t � } | j d t � } t | � | f } | t k s^ t r| j d i  � } t | | � r� | t | <| rt | j | �  qqd | k s� d | k s� d | k s� d | k r� | d	 | d
 | | � t | <q| d | d
 | | � t | <n  t | s2t	 d d | � n  x |  d D] } | j
 | � q=Wt | j | � S(   s�   
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    i    t   template_adaptert   template_lookupt   template_settingss   
t   {t   %t   $Rz  R�  R�   i�  s   Template (%s) not foundi   N(   Rg   R�   R�  t   TEMPLATE_PATHt   idt	   TEMPLATESR�   R>   R�   R�  RJ  R�  (   R�   R.  R�  t   adapterR�  t   tplidR�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRb    s$    
 0
 R  c         �  s   �  � f d �  } | S(   s�   Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    c         �  s(   t  j �  � � �  � f d �  � } | S(   Nc          �  sg   � |  | �  } t  | t t f � rJ �  j �  } | j | � t � | � S| d  k rc t � �  � S| S(   N(   R>   R]   R9   R�  RJ  Rb  Rg   (   R�   R.  t   resultt   tplvars(   R�  Rf   t   tpl_name(    s&   /home/lgardner/git/professor/bottle.pyRP   +  s    (   RM   R  (   Rf   RP   (   R�  R+  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  *  s    $
(    (   R+  R�  R0  (    (   R�  R+  s&   /home/lgardner/git/professor/bottle.pyR?     s    
s   ./s   ./views/s	   ../views/s   I'm a teapoti�  s   Unprocessable Entityi�  s   Precondition Requiredi�  s   Too Many Requestsi�  s   Request Header Fields Too Largei�  s   Network Authentication Requiredi�  c         c�  s+   |  ]! \ } } | d  | | f f Vq d S(   s   %d %sN(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>S  s    s�  
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
RL  Rv  t	   localhostR�  t   ]s   []R�  R�  R  R�  R�   R�  (  Rp   t
   __future__R    t
   __author__R�  t   __license__RK   t   optparseR   t   _cmd_parsert
   add_optiont   _optt
   parse_argst   _cmd_optionst	   _cmd_argsR  R�  t   gevent.monkeyR   t   monkeyt	   patch_allR�  R�  t   email.utilsR�  RM   R�  RG  R:  R�  R�  R�   R�  R   R�  R4  R+  RT   R   R   R)  R   R   R;  R   R   t   inspectR   t   unicodedataR   t
   simplejsonR   R   R   R.   R   R�  t   django.utils.simplejsont   version_infot   pyR  t   py25R�  R   R   R   R"   R�  RA  R�  t   http.clientt   clientt   httplibt   _threadR�  t   urllib.parseR#   R$   R�  R%   R&   R�  R'   R�  R  t   http.cookiesR*   t   collectionsR+   R9   R�  t   ioR,   t   configparserR-   R�   R?  R?   R�  RI  R�  R6   R5   t   urlparset   urllibt   Cookiet   cPickleR7   R8   R�   RU   RV   t   UserDictR:   RA   R�  R�   RC   R/   R�  RG   RH   RN   Rq   RY   R^   R�  R_   Rr   Rt   Rm  Rv   Rw   Rx   Ry   Rz   R{   R�   R�   R�   R  R�  R  R  R  Rg   R6  R7  R8  R�  t   ResponseR9  R�   R<  R!  R"  R@  RU  R�  R  R�  R]   R�   R[   R�  Rt  Rw  R  R�  R�  R�  R�  R�  R�   R�  R#  R"  R�  R�  R�  R�  R&  R�  R�  R�  R�  RX  R8  R  R  RA  R�   RZ  R\  R^  R�   RE  R/  R   RL  R�   R  R  R
  R  R$  R/  R2  R8  R;  RE  RQ  RV  Re  Rh  Rn  Rq  Rx  R{  R~  R�  R�  RW  R�  R�  RN  R�  R�  R�  R�  R�  R�  R�  R�  R�  R�  Rb  t   mako_templatet   cheetah_templatet   jinja2_templateR?  t	   mako_viewt   cheetah_viewt   jinja2_viewR$  R&  R�   R�  t	   responsest
   HTTP_CODESR	  R  Rc  R7  Rh  R5  R�   R�  R�  RI  R�  t   optR�   Rt  t   versionR�  t
   print_helpR�   R)  RF  R�   Rg  R�  R�  t   rfindRM  RP  R�   R�  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyt   <module>   s  	 �   		.	"									�w� �� �	�
$I/2�VH
Q				
				
					
	


		Z1OH�			
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
�
��Xc           @�  s�  d  Z  d d l m Z d Z d Z d Z e d k r5d d l m Z e d d	 � Z	 e	 j
 Z e d
 d d d d �e d d d d d d �e d d d d d d �e d d d d d d �e d d d d d �e d d d d d  �e	 j �  \ Z Z e j oe j j d! � r2d d" l Z e j j �  n  n  d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l Z d d" l  Z  d d" l! Z! d d" l" Z" d d" l# Z# d d" l$ Z$ d d" l% Z% d d# l& m' Z( m& Z& m) Z) d d$ l" m* Z* d d% l+ m, Z, m- Z- d d& l. m/ Z/ d d' l0 m1 Z1 y d d( l2 m3 Z4 m5 Z6 Wn| e7 k
 r�y d d( l8 m3 Z4 m5 Z6 WnN e7 k
 r�y d d( l9 m3 Z4 m5 Z6 Wn  e7 k
 r�d) �  Z4 e4 Z6 n Xn Xn Xe! j: Z; e; d* d+ d+ f k Z< e; d, d- d+ f k  Z= d* d. d+ f e; k oLd* d, d+ f k  n Z> d/ �  Z? y" e! j@ jA e! jB jA f \ ZC ZD Wn# eE k
 r�d0 �  ZC d1 �  ZD n Xe< r�d d" lF jG ZH d d" lI ZJ d d2 lK mL ZL mM ZN d d3 lK mO ZO mP ZQ mR ZS e jT eS d4 d5 �ZS d d6 lU mV ZV d d7 lW mX ZY d d" lZ ZZ d d8 l[ m\ Z\ d d9 l] m^ Z^ e_ Z` e_ Za d: �  Zb d; �  Zc ed Ze d< �  Zf nd d" lH ZH d d" lJ ZJ d d2 lg mL ZL mM ZN d d3 lh mO ZO mP ZQ mR ZS d d6 li mV ZV d d= l me Ze d d" lj ZZ d d> lk mk Z\ d d? l^ ml Z^ e= rZd@ Zm e% jn em eo � d dA lp mY ZY dB �  Zq e_ Zr n d d7 lW mX ZY ea Za e6 Zb es et dC dD dE � � dF dG � Zu dF dH dI � Zv e< r�ev n eu Zw e> r�d dJ l[ mx Zx dK ex f dL �  �  YZy n  dM �  Zz e{ dN � Z| dO �  Z} dP e~ f dQ �  �  YZ dR e~ f dS �  �  YZ� dT e~ f dU �  �  YZ� dV e� f dW �  �  YZ� dX e� f dY �  �  YZ� dZ e� f d[ �  �  YZ� d\ e� f d] �  �  YZ� d^ e� f d_ �  �  YZ� d` e� f da �  �  YZ� db �  Z� dc e~ f dd �  �  YZ� de e~ f df �  �  YZ� dg e~ f dh �  �  YZ� di e~ f dj �  �  YZ� dk �  Z� dl e~ f dm �  �  YZ� dn e~ f do �  �  YZ� e� dp � Z� dq e� f dr �  �  YZ� ds e� f dt �  �  YZ� e� Z� e� Z� du e� e� f dv �  �  YZ� dw e� f dx �  �  YZ� dy e� f dz �  �  YZ� d{ e~ f d| �  �  YZ� d} e~ f d~ �  �  YZ� d e~ f d� �  �  YZ� d� eY f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� eY f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e~ f d� �  �  YZ� d� d� d� � Z� e� d� � Z� d� d� d� � Z� d� e{ d� d� � Z� e� d� � Z� d� �  Z� d� �  Z� d� �  Z� d+ d� � Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d� �  Z� d. d� � Z� d� d� d� � Z� d� �  Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� e� d� � Z� d� e~ f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e� f d� �  �  YZ� i e� d� 6e� d� 6e� d 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d� 6e� d! 6e� d� 6e� d� 6e� d� 6e� d� 6Z� d� �  Z� d� �  Z� e� Z� e� d d� d� d. e{ e{ e� e� d� �	 Z� d� e# j� f d� �  �  YZ� d� e� f d� �  �  YZ� d� e~ f d� �  �  YZ� d� e� f d� �  �  YZ� d e� f d�  �  YZ� de� f d�  �  YZ� de� f d�  �  YZ� de� f d�  �  YZ� de~ f d	�  �  YZ� d
�  Z� e jT e� de� �Z� e jT e� de� �Z� e jT e� de� �Z� d�  Z� e jT e� de� �Z� e jT e� de� �Z� e jT e� de� �Z� dddg Z� i  Z� e{ a� e{ a� eH j� Z� de� d<de� d<de� d<de� d<de� d<de� d<e� d�  e� j� �  D� � Z� de Z� e� �  Z� e� �  Z� e# j� �  Z� e� �  Z Ze j�  e� e d k rdn e dd � jZe d k r�e e e	 f \ ZZZejrueC d!e � e! j	d+ � n  er�ej
�  eD d"� e! j	d. � n  e! jjd+ d#� e! jjd$e! jd � ejp�d%d� f \ ZZd&ek oejd'� ejd&� k  r-ejd&d. � \ ZZn  ejd(� Ze� ed+ d)ed*ee� d+ej d,ejd-ejd.ej� �n  d" S(/  s�  
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2013, Marcel Hellkamp.
License: MIT (see LICENSE for details)
i����(   t   with_statements   Marcel Hellkamps   0.12.9t   MITt   __main__(   t   OptionParsert   usages)   usage: %prog [options] package.module:apps	   --versiont   actiont
   store_truet   helps   show version number.s   -bs   --bindt   metavart   ADDRESSs   bind socket to ADDRESS.s   -ss   --servert   defaultt   wsgirefs   use SERVER as backend.s   -ps   --plugint   appends   install additional plugin/s.s   --debugs   start server in debug mode.s   --reloads   auto-reload on file changes.t   geventN(   t   datet   datetimet	   timedelta(   t   TemporaryFile(   t
   format_exct	   print_exc(   t
   getargspec(   t	   normalize(   t   dumpst   loadsc         C�  s   t  d � � d  S(   Ns/   JSON support requires Python 2.6 or simplejson.(   t   ImportError(   t   data(    (    s&   /home/lgardner/git/professor/bottle.pyt
   json_dumps6   s    i   i    i   i   i   c           C�  s   t  j �  d S(   Ni   (   t   syst   exc_info(    (    (    s&   /home/lgardner/git/professor/bottle.pyt   _eE   s    c         C�  s   t  j j |  � S(   N(   R   t   stdoutt   write(   t   x(    (    s&   /home/lgardner/git/professor/bottle.pyt   <lambda>L   s    c         C�  s   t  j j |  � S(   N(   R   t   stderrR   (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   M   s    (   t   urljoint   SplitResult(   t	   urlencodet   quotet   unquotet   encodingt   latin1(   t   SimpleCookie(   t   MutableMapping(   t   BytesIO(   t   ConfigParserc         C�  s   t  t |  � � S(   N(   t   json_ldst   touni(   t   s(    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]   s    c         C�  s   t  |  d � S(   Nt   __call__(   t   hasattr(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ^   s    c          G�  s%   |  d |  d � j  |  d � � d  S(   Ni    i   i   (   t   with_traceback(   t   a(    (    s&   /home/lgardner/git/professor/bottle.pyt   _raise`   s    (   t   imap(   t   StringIO(   t   SafeConfigParsers?   Python 2.5 support may be dropped in future versions of Bottle.(   t	   DictMixinc         C�  s
   |  j  �  S(   N(   t   next(   t   it(    (    s&   /home/lgardner/git/professor/bottle.pyR:   o   s    s&   def _raise(*a): raise a[0], a[1], a[2]s   <py3fix>t   exect   utf8c         C�  s&   t  |  t � r |  j | � St |  � S(   N(   t
   isinstancet   unicodet   encodet   bytes(   R0   t   enc(    (    s&   /home/lgardner/git/professor/bottle.pyt   tobx   s    t   strictc         C�  s)   t  |  t � r |  j | | � St |  � S(   N(   R>   RA   t   decodeR?   (   R0   RB   t   err(    (    s&   /home/lgardner/git/professor/bottle.pyR/   z   s    (   t   TextIOWrappert   NCTextIOWrapperc           B�  s   e  Z d  �  Z RS(   c         C�  s   d  S(   N(    (   t   self(    (    s&   /home/lgardner/git/professor/bottle.pyt   close�   s    (   t   __name__t
   __module__RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRH   �   s   c         O�  s2   y t  j |  | | | � Wn t k
 r- n Xd  S(   N(   t	   functoolst   update_wrappert   AttributeError(   t   wrappert   wrappedR4   t   ka(    (    s&   /home/lgardner/git/professor/bottle.pyRN   �   s      c         C�  s   t  j |  t d d �d  S(   Nt
   stackleveli   (   t   warningst   warnt   DeprecationWarning(   t   messaget   hard(    (    s&   /home/lgardner/git/professor/bottle.pyt   depr�   s    c         C�  s:   t  |  t t t t f � r% t |  � S|  r2 |  g Sg  Sd  S(   N(   R>   t   tuplet   listt   sett   dict(   R   (    (    s&   /home/lgardner/git/professor/bottle.pyt   makelist�   s
     
 t   DictPropertyc           B�  sA   e  Z d  Z d e d � Z d �  Z d �  Z d �  Z d �  Z	 RS(   s=    Property that maps to a key in a local dict-like attribute. c         C�  s!   | | | |  _  |  _ |  _ d  S(   N(   t   attrt   keyt	   read_only(   RI   R`   Ra   Rb   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __init__�   s    c         C�  s9   t  j |  | d g  �| |  j p( | j |  _ |  _ |  S(   Nt   updated(   RM   RN   Ra   RK   t   getter(   RI   t   func(    (    s&   /home/lgardner/git/professor/bottle.pyR1   �   s    c         C�  sV   | d  k r |  S|  j t | |  j � } } | | k rN |  j | � | | <n  | | S(   N(   t   NoneRa   t   getattrR`   Re   (   RI   t   objt   clsRa   t   storage(    (    s&   /home/lgardner/git/professor/bottle.pyt   __get__�   s      c         C�  s5   |  j  r t d � � n  | t | |  j � |  j <d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   t   value(    (    s&   /home/lgardner/git/professor/bottle.pyt   __set__�   s    	 c         C�  s2   |  j  r t d � � n  t | |  j � |  j =d  S(   Ns   Read-Only property.(   Rb   RO   Rh   R`   Ra   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   __delete__�   s    	 N(
   RK   RL   t   __doc__Rg   t   FalseRc   R1   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR_   �   s   			t   cached_propertyc           B�  s    e  Z d  Z d �  Z d �  Z RS(   s�    A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. c         C�  s   t  | d � |  _ | |  _ d  S(   NRp   (   Rh   Rp   Rf   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �   s    c         C�  s4   | d  k r |  S|  j | � } | j |  j j <| S(   N(   Rg   Rf   t   __dict__RK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �   s      (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRr   �   s   	t   lazy_attributec           B�  s    e  Z d  Z d �  Z d �  Z RS(   s4    A property that caches itself to the class object. c         C�  s#   t  j |  | d g  �| |  _ d  S(   NRd   (   RM   RN   Re   (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �   s    c         C�  s&   |  j  | � } t | |  j | � | S(   N(   Re   t   setattrRK   (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �   s    (   RK   RL   Rp   Rc   Rl   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt   �   s   	t   BottleExceptionc           B�  s   e  Z d  Z RS(   s-    A base class for exceptions used by bottle. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRv   �   s   t
   RouteErrorc           B�  s   e  Z d  Z RS(   s9    This is a base class for all routing related exceptions (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw   �   s   t
   RouteResetc           B�  s   e  Z d  Z RS(   sf    If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx   �   s   t   RouterUnknownModeErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRy   �   s    t   RouteSyntaxErrorc           B�  s   e  Z d  Z RS(   s@    The route parser found something not supported by this router. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRz   �   s   t   RouteBuildErrorc           B�  s   e  Z d  Z RS(   s    The route could not be built. (   RK   RL   Rp   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{   �   s   c         C�  s&   d |  k r |  St  j d d �  |  � S(   s^    Turn all capturing groups in a regular expression pattern into
        non-capturing groups. t   (s   (\\*)(\(\?P<[^>]+>|\((?!\?))c         S�  s7   t  |  j d � � d r& |  j d � S|  j d � d S(   Ni   i   i    s   (?:(   t   lent   group(   t   m(    (    s&   /home/lgardner/git/professor/bottle.pyR!   �   s    (   t   ret   sub(   t   p(    (    s&   /home/lgardner/git/professor/bottle.pyt   _re_flatten�   s     	t   Routerc           B�  st   e  Z d  Z d Z d Z d Z e d � Z d �  Z e	 j
 d � Z d �  Z d d � Z d	 �  Z d
 �  Z d �  Z RS(   sA   A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    s   [^/]+R�   ic   c         �  sz   g  �  _  i  �  _ i  �  _ i  �  _ i  �  _ i  �  _ | �  _ i �  f d �  d 6d �  d 6d �  d 6d �  d 6�  _ d  S(	   Nc         �  s   t  |  p �  j � d  d  f S(   N(   R�   t   default_patternRg   (   t   conf(   RI   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d t  d �  f S(   Ns   -?\d+c         S�  s   t  t |  � � S(   N(   t   strt   int(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d t  d �  f S(   Ns   -?[\d.]+c         S�  s   t  t |  � � S(   N(   R�   t   float(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    (   R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    R�   c         S�  s   d S(   Ns   .+?(   s   .+?NN(   Rg   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!      s    t   path(   t   rulest   _groupst   buildert   statict   dyna_routest   dyna_regexest   strict_ordert   filters(   RI   RD   (    (   RI   s&   /home/lgardner/git/professor/bottle.pyRc     s    							

c         C�  s   | |  j  | <d S(   s�    Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. N(   R�   (   RI   t   nameRf   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   add_filter"  s    s�   (\\*)(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)(?::((?:\\.|[^\\>]+)+)?)?)?>))c   	      c�  s?  d \ } } x� |  j  j | � D]� } | | | | j �  !7} | j �  } t | d � d r� | | j d � t | d � 7} | j �  } q n  | r� | d  d  f Vn  | d d  k r� | d d !n
 | d d !\ } } } | | p� d | p� d  f V| j �  d } } q W| t | � k s"| r;| | | d  d  f Vn  d  S(	   Ni    t    i   i   i   i   R
   (   i    R�   (   t   rule_syntaxt   finditert   startt   groupsR}   R~   t   endRg   (	   RI   t   rulet   offsett   prefixt   matcht   gR�   t   filtrR�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _itertokens-  s    !3c         �  s�  d } g  } d } g  �  g  } t  }	 x|  j | � D]\ }
 } } | rt }	 | d k rg |  j } n  |  j | | � \ } } } |
 s� | d | 7} d | }
 | d 7} n! | d |
 | f 7} | j |
 � | r� �  j |
 | f � n  | j |
 | p� t f � q4 |
 r4 | t j |
 � 7} | j d |
 f � q4 q4 W| |  j
 | <| r]| |  j
 | <n  |	 r�|  j r�|  j j | i  � | d f |  j | |  j | � <d Sy  t j d	 | � } | j � Wn- t j k
 r�t d
 | t �  f � � n X�  r�  � f d �  } n! | j r*� f d �  } n d } t | � } | | | | f } | | f |  j k r�t r�d } t j | | | f t � n  | |  j | |  j | | f <n@ |  j j | g  � j | � t |  j | � d |  j | | f <|  j | � d S(   s<    Add a new rule or replace the target for an existing rule. i    R�   R
   s   (?:%s)s   anon%di   s
   (?P<%s>%s)Ns   ^(%s)$s   Could not add Route: %s (%s)c         �  sh   � |  � j  �  } xO �  D]G \ } } y | | | � | | <Wq t k
 r_ t d d � � q Xq W| S(   Ni�  s   Path has wrong format.(   t	   groupdictt
   ValueErrort	   HTTPError(   R�   t   url_argsR�   t   wildcard_filter(   R�   t   re_match(    s&   /home/lgardner/git/professor/bottle.pyt   getargsh  s    c         �  s   �  |  � j  �  S(   N(   R�   (   R�   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyR�   q  s    s3   Route <%s %s> overwrites a previously defined route(   t   TrueR�   Rq   t   default_filterR�   R   R�   R�   t   escapeRg   R�   R�   R�   t
   setdefaultt   buildt   compileR�   t   errorRz   R   t
   groupindexR�   R�   t   DEBUGRT   RU   t   RuntimeWarningR�   R}   t   _compile(   RI   R�   t   methodt   targetR�   t   anonst   keyst   patternR�   t	   is_staticRa   t   modeR�   t   maskt	   in_filtert
   out_filtert
   re_patternR�   t   flatpatt
   whole_rulet   msg(    (   R�   R�   s&   /home/lgardner/git/professor/bottle.pyt   add>  sf     
   	!$c         C�  s�   |  j  | } g  } |  j | <|  j } x� t d t | � | � D]� } | | | | !} d �  | D� } d j d �  | D� � } t j | � j } g  | D] \ } } }	 }
 |	 |
 f ^ q� } | j	 | | f � q@ Wd  S(   Ni    c         s�  s!   |  ] \ } } } } | Vq d  S(   N(    (   t   .0t   _R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    t   |c         s�  s   |  ] } d  | Vq d S(   s   (^%s$)N(    (   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    (
   R�   R�   t   _MAX_GROUPS_PER_PATTERNt   rangeR}   t   joinR�   R�   R�   R   (   RI   R�   t	   all_rulest
   comborulest	   maxgroupsR    t   somet   combinedR�   R�   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	+c   
      O�  s�   |  j  j | � } | s* t d | � � n  y� x( t | � D] \ } } | | d | <q: Wd j g  | D]- \ } } | r� | | j | � � n | ^ qe � }	 | s� |	 S|	 d t | � SWn+ t k
 r� t d t �  j	 d � � n Xd S(   s2    Build an URL by filling the wildcards in a rule. s   No route with that name.s   anon%dR�   t   ?s   Missing URL argument: %ri    N(
   R�   t   getR{   t	   enumerateR�   t   popR%   t   KeyErrorR   t   args(
   RI   t   _nameR�   t   queryR�   t   iRm   t   nt   ft   url(    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s      C c         C�  s<  | d j  �  } | d p d } d } | d k rG d | d d g } n d | d g } x� | D]� } | |  j k r� | |  j | k r� |  j | | \ } } | | r� | | � n i  f S| |  j k r] xc |  j | D]Q \ } }	 | | � }
 |
 r� |	 |
 j d \ } } | | r| | � n i  f Sq� Wq] q] Wt g  � } t | � } x> t |  j � | D]) } | |  j | k r]| j | � q]q]Wx_ t |  j � | | D]F } x= |  j | D]. \ } }	 | | � }
 |
 r�| j | � q�q�Wq�W| rd	 j t | � � } t	 d
 d d | �� n  t	 d d t
 | � � � d S(   sD    Return a (target, url_agrs) tuple or raise HTTPError(400/404/405). t   REQUEST_METHODt	   PATH_INFOt   /t   HEADt   PROXYt   GETt   ANYi   t   ,i�  s   Method not allowed.t   Allowi�  s   Not found: N(   t   upperRg   R�   R�   t	   lastindexR\   R�   R�   t   sortedR�   t   repr(   RI   t   environt   verbR�   R�   t   methodsR�   R�   R�   R�   R�   t   allowedt   nocheckt   allow_header(    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s<    "'N(   RK   RL   Rp   R�   R�   R�   Rq   Rc   R�   R�   R�   R�   R�   Rg   R�   R�   R�   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �   s   
		F		t   Routec           B�  s�   e  Z d  Z d d d d � Z d �  Z e d �  � Z d �  Z d �  Z	 e
 d �  � Z d �  Z d �  Z d	 �  Z d
 �  Z d d � Z d �  Z RS(   s�    This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    c   	      K�  sp   | |  _  | |  _ | |  _ | |  _ | p- d  |  _ | p< g  |  _ | pK g  |  _ t �  j	 | d t
 �|  _ d  S(   Nt   make_namespaces(   t   appR�   R�   t   callbackRg   R�   t   pluginst   skiplistt
   ConfigDictt	   load_dictR�   t   config(	   RI   R�   R�   R�   R�   R�   R�   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    				c         O�  s   t  d � |  j | | �  S(   Ns�   Some APIs changed to return Route() instances instead of callables. Make sure to use the Route.call method and not to call Route instances directly.(   RY   t   call(   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    
c         C�  s
   |  j  �  S(   s�    The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests.(   t   _make_callback(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   |  j  j d d � d S(   sk    Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. R�   N(   Rs   R�   Rg   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   reset�  s    c         C�  s   |  j  d S(   s:    Do all on-demand work immediately (useful for debugging).N(   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   prepare�  s    c         C�  sY   t  d � t d |  j d |  j d |  j d |  j d |  j d |  j d |  j d	 |  j	 � S(
   Ns=   Switch to Plugin API v2 and access the Route object directly.R�   R�   R�   R�   R�   R�   t   applyt   skip(
   RY   R]   R�   R�   R�   R�   R�   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _context�  s    
!c         c�  s�   t  �  } x� t |  j j |  j � D]� } t |  j k r< Pn  t | d t � } | ru | |  j k s# | | k ru q# n  | |  j k s# t | � |  j k r� q# n  | r� | j	 | � n  | Vq# Wd S(   s)    Yield all Plugins affecting this route. R�   N(
   R\   t   reversedR�   R�   R�   R�   Rh   Rq   t   typeR�   (   RI   t   uniqueR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   all_plugins�  s    	  ! $  c         C�  s�   |  j  } x� |  j �  D]� } ya t | d � rp t | d d � } | d k rR |  n |  j } | j | | � } n | | � } Wn t k
 r� |  j �  SX| |  j  k	 r t | |  j  � q q W| S(   NR�   t   apii   (	   R�   R   R2   Rh   R�   R�   Rx   R�   RN   (   RI   R�   t   pluginR  t   context(    (    s&   /home/lgardner/git/professor/bottle.pyR�   	  s    	c         C�  sx   |  j  } t | t r d n d | � } t r3 d n d } x8 t | | � rs t | | � rs t | | � d j } q< W| S(   sq    Return the callback. If the callback is a decorated function, try to
            recover the original function. t   __func__t   im_funct   __closure__t   func_closurei    (   R�   Rh   t   py3kR2   t   cell_contents(   RI   Rf   t   closure_attr(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_undecorated_callback  s    	!c         C�  s   t  |  j �  � d S(   s�    Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. i    (   R   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   get_callback_args#  s    c         C�  s8   x1 |  j  |  j j f D] } | | k r | | Sq W| S(   sp    Lookup a config field and return its value, first checking the
            route.config, then route.app.config.(   R�   R�   t   conifg(   RI   Ra   R
   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_config)  s     c         C�  s#   |  j  �  } d |  j |  j | f S(   Ns
   <%s %r %r>(   R  R�   R�   (   RI   t   cb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __repr__0  s    N(   RK   RL   Rp   Rg   Rc   R1   Rr   R�   R�   R�   t   propertyR�   R   R�   R  R  R  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s   						
	t   Bottlec           B�  s[  e  Z d  Z e e d � Z e d d � Z d& Z d Z e	 d �  � Z
 d �  Z d	 �  Z d
 �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d' d � Z d �  Z d �  Z d �  Z d �  Z d �  Z d' d d' d' d' d' d � Z d' d d � Z d' d d � Z d' d d � Z d' d d � Z d d  � Z d! �  Z  d" �  Z! d' d# � Z" d$ �  Z# d% �  Z$ RS((   s^   Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    c         C�  s�   t  �  |  _ t j |  j d � |  j _ |  j j d d t � |  j j d d t � | |  j d <| |  j d <t �  |  _	 g  |  _
 t �  |  _ i  |  _ g  |  _ |  j d r� |  j t �  � n  |  j t �  � d  S(   NR�   t   autojsont   validatet   catchall(   R�   R�   RM   t   partialt   trigger_hookt
   _on_changet   meta_sett   boolt   ResourceManagert	   resourcest   routesR�   t   routert   error_handlerR�   t   installt
   JSONPlugint   TemplatePlugin(   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   G  s    			R�   R  t   before_requestt   after_requestt	   app_resetc         C�  s   t  d �  |  j D� � S(   Nc         s�  s   |  ] } | g  f Vq d  S(   N(    (   R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>f  s    (   R]   t   _Bottle__hook_names(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hooksd  s    c         C�  sA   | |  j  k r) |  j | j d | � n |  j | j | � d S(   s�   Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        i    N(   t   _Bottle__hook_reversedR'  t   insertR   (   RI   R�   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   add_hookh  s    c         C�  s>   | |  j  k r: | |  j  | k r: |  j  | j | � t Sd S(   s     Remove a callback from a hook. N(   R'  t   removeR�   (   RI   R�   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyt   remove_hookx  s    "c         O�  s(   g  |  j  | D] } | | | �  ^ q S(   s.    Trigger a hook and return a list of results. (   R'  (   RI   t   _Bottle__nameR�   t   kwargst   hook(    (    s&   /home/lgardner/git/professor/bottle.pyR  ~  s    c         �  s   �  � f d �  } | S(   se    Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details.c         �  s   � j  �  |  � |  S(   N(   R*  (   Rf   (   R�   RI   (    s&   /home/lgardner/git/professor/bottle.pyt	   decorator�  s    (    (   RI   R�   R0  (    (   R�   RI   s&   /home/lgardner/git/professor/bottle.pyR/  �  s    c         �  s  t  �  t � r t d t � n  g  | j d � D] } | r/ | ^ q/ } | s\ t d � � n  t | � � �  � f d �  } | j d t � | j d d � | j d i | d	 6�  d
 6� | | d <|  j d d j	 | � | � | j
 d � s|  j d d j	 | � | � n  d S(   s�   Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        s*   Parameter order of Bottle.mount() changed.R�   s   Empty path prefix.c          �  s�   z~ t  j � � t g  � �  d  �  f d � }  � t  j |  � } | rg �  j rg t j �  j | � } n  | ps �  j �  _ �  SWd  t  j � � Xd  S(   Nc         �  s[   | r! z t  | �  Wd  d  } Xn  |  �  _ x$ | D] \ } } �  j | | � q1 W�  j j S(   N(   R5   Rg   t   statust
   add_headert   bodyR   (   R1  t
   headerlistR   R�   Rm   (   t   rs(    s&   /home/lgardner/git/professor/bottle.pyt   start_response�  s    
	 (   t   requestt
   path_shiftt   HTTPResponseRg   R�   R3  t	   itertoolst   chain(   R6  R3  (   R�   t
   path_depth(   R5  s&   /home/lgardner/git/professor/bottle.pyt   mountpoint_wrapper�  s    	 R�   R�   R�   t
   mountpointR�   R�   R�   s   /%s/<:re:.*>N(   R>   t
   basestringRY   R�   t   splitR�   R}   R�   t   routeR�   t   endswith(   RI   R�   R�   t   optionsR�   t   segmentsR=  (    (   R�   R<  s&   /home/lgardner/git/professor/bottle.pyt   mount�  s    ( 
c         C�  s=   t  | t � r | j } n  x | D] } |  j | � q" Wd S(   s�    Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. N(   R>   R  R  t	   add_route(   RI   R  RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   merge�  s    c         C�  si   t  | d � r | j |  � n  t | � rK t  | d � rK t d � � n  |  j j | � |  j �  | S(   s�    Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        t   setupR�   s.   Plugins must be callable or implement .apply()(   R2   RH  t   callablet	   TypeErrorR�   R   R�   (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s     
c         C�  s�   g  | } } x� t  t |  j � � d d d � D]� \ } } | t k s~ | | k s~ | t | � k s~ t | d t � | k r0 | j | � |  j | =t | d � r� | j �  q� q0 q0 W| r� |  j	 �  n  | S(   s)   Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. Ni����R�   RJ   (
   R[   R�   R�   R�   R�   Rh   R   R2   RJ   R�   (   RI   R  t   removedR+  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   uninstall�  s    /*
  c         C�  s�   | d k r |  j } n+ t | t � r3 | g } n |  j | g } x | D] } | j �  qJ Wt r� x | D] } | j �  qk Wn  |  j d � d S(   s�    Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. R%  N(   Rg   R  R>   R�   R�   R�   R�   R  (   RI   RA  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s        c         C�  s=   x- |  j  D]" } t | d � r
 | j �  q
 q
 Wt |  _ d S(   s2    Close the application and all installed plugins. RJ   N(   R�   R2   RJ   R�   t   stopped(   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s     c         K�  s   t  |  | � d S(   s-    Calls :func:`run` with the same parameters. N(   t   run(   RI   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    c         C�  s   |  j  j | � S(   s�    Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match.(   R  R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         K�  sV   t  j j d d � j d � d } |  j j | | � j d � } t t d | � | � S(   s,    Return a string that matches a named route t   SCRIPT_NAMER�   R�   (   R7  R�   R�   t   stripR  R�   t   lstripR#   (   RI   t	   routenamet   kargst
   scriptnamet   location(    (    s&   /home/lgardner/git/professor/bottle.pyt   get_url�  s    "c         C�  sL   |  j  j | � |  j j | j | j | d | j �t rH | j �  n  d S(   sS    Add a route object, but do not change the :data:`Route.app`
            attribute.R�   N(	   R  R   R  R�   R�   R�   R�   R�   R�   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyRF    s    % R�   c   	      �  si   t  � � r d � � } n  t | � � t | � � �  � � � � � � f d �  } | re | | � S| S(   s   A decorator to bind a function to a request URL. Example::

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
        c         �  s�   t  |  t � r t |  � }  n  xz t � � p6 t |  � D]` } xW t � � D]I } | j �  } t � | | |  d � d � d � �  �} � j | � qJ Wq7 W|  S(   NR�   R�   R�   (   R>   R?  t   loadR^   t   yieldroutesR�   R�   RF  (   R�   R�   R�   RA  (   R�   R�   R�   R�   R�   RI   R�   (    s&   /home/lgardner/git/professor/bottle.pyR0  &  s     N(   RI  Rg   R^   (	   RI   R�   R�   R�   R�   R�   R�   R�   R0  (    (   R�   R�   R�   R�   R�   RI   R�   s&   /home/lgardner/git/professor/bottle.pyRA    s     !
c         K�  s   |  j  | | | � S(   s    Equals :meth:`route`. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   2  s    t   POSTc         K�  s   |  j  | | | � S(   s8    Equals :meth:`route` with a ``POST`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   post6  s    t   PUTc         K�  s   |  j  | | | � S(   s7    Equals :meth:`route` with a ``PUT`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   put:  s    t   DELETEc         K�  s   |  j  | | | � S(   s:    Equals :meth:`route` with a ``DELETE`` method parameter. (   RA  (   RI   R�   R�   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete>  s    i�  c         �  s   �  � f d �  } | S(   s<    Decorator: Register an output handler for a HTTP error codec         �  s   |  � j  t �  � <|  S(   N(   R  R�   (   t   handler(   t   codeRI   (    s&   /home/lgardner/git/professor/bottle.pyRP   D  s    (    (   RI   R`  RP   (    (   R`  RI   s&   /home/lgardner/git/professor/bottle.pyR�   B  s    c         C�  s   t  t t d | �� S(   Nt   e(   RC   t   templatet   ERROR_PAGE_TEMPLATE(   RI   t   res(    (    s&   /home/lgardner/git/professor/bottle.pyt   default_error_handlerI  s    c         C�  s�  | d } | d <t  rY y  | j d � j d � | d <WqY t k
 rU t d d � SXn  y� |  | d <t j | � t j �  zT |  j d � |  j	 j
 | � \ } } | | d	 <| | d
 <| | d <| j | �  SWd  |  j d � XWn� t k
 r� t �  St k
 r| j �  |  j | � St t t f k
 r:�  nM t k
 r�|  j sV�  n  t �  } | d j | � t d d t �  | � SXd  S(   NR�   s   bottle.raw_pathR)   R=   i�  s#   Invalid path string. Expected UTF-8s
   bottle.appR#  s   route.handles   bottle.routes   route.url_argsR$  s   wsgi.errorsi�  s   Internal Server Error(   R  R@   RE   t   UnicodeErrorR�   R7  t   bindt   responseR  R  R�   R�   R9  R   Rx   R�   t   _handlet   KeyboardInterruptt
   SystemExitt   MemoryErrort	   ExceptionR  R   R   (   RI   R�   R�   RA  R�   t
   stacktrace(    (    s&   /home/lgardner/git/professor/bottle.pyRi  L  s>     





	 	c         C�  s$  | s# d t  k r d t  d <n  g  St | t t f � rn t | d t t f � rn | d d d !j | � } n  t | t � r� | j t  j � } n  t | t � r� d t  k r� t	 | � t  d <n  | g St | t
 � r| j t  � |  j j | j |  j � | � } |  j | � St | t � r=| j t  � |  j | j � St | d � r�d t j k rlt j d | � St | d � s�t | d � r�t | � Sn  y5 t | � } t | � } x | s�t | � } q�WWn� t k
 r�|  j d � St k
 rt �  } nW t t t f k
 r�  n; t k
 rY|  j s;�  n  t
 d d	 t �  t  �  � } n Xt | t � rv|  j | � St | t � r�t! j" | g | � } n_ t | t � r�d
 �  } t# | t! j" | g | � � } n& d t$ | � } |  j t
 d | � � St | d � r t% | | j& � } n  | S(   s�    Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        s   Content-Lengthi    t   reads   wsgi.file_wrapperRJ   t   __iter__R�   i�  s   Unhandled exceptionc         S�  s   |  j  t j � S(   N(   R@   Rh  t   charset(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   �  s    s   Unsupported response type: %s('   Rh  R>   RZ   R[   RA   R?   R�   R@   Rq  R}   R�   R�   R  R�   t   status_codeRe  t   _castR9  R3  R2   R7  R�   t   WSGIFileWrappert   iterR:   t   StopIterationR   Rj  Rk  Rl  Rm  R  R   R:  R;  R6   R�   t
   _closeiterRJ   (   RI   t   outt   peekt   ioutt   firstt   new_itert   encoderR�   (    (    s&   /home/lgardner/git/professor/bottle.pyRs  o  sh    !		 	!c         C�  sE  yw |  j  |  j | � � } t j d k s: | d d k r_ t | d � rV | j �  n  g  } n  | t j t j � | SWn� t t	 t
 f k
 r� �  n� t k
 r@|  j s� �  n  d t | j d	 d
 � � } t r| d t t t �  � � t t �  � f 7} n  | d j | � d g } | d | t j �  � t | � g SXd S(   s    The bottle WSGI-interface. id   ie   i�   i0  R�   R�   RJ   s4   <h1>Critical error while processing request: %s</h1>R�   R�   sD   <h2>Error:</h2>
<pre>
%s
</pre>
<h2>Traceback:</h2>
<pre>
%s
</pre>
s   wsgi.errorss   Content-Types   text/html; charset=UTF-8s   500 INTERNAL SERVER ERRORN(   id   ie   i�   i0  (   s   Content-Types   text/html; charset=UTF-8(   Rs  Ri  Rh  t   _status_codeR2   RJ   t   _status_lineR4  Rj  Rk  Rl  Rm  R  t   html_escapeR�   R�   R�   R   R   R   R   R   RC   (   RI   R�   R6  Rx  RF   t   headers(    (    s&   /home/lgardner/git/professor/bottle.pyt   wsgi�  s.     		 )	c         C�  s   |  j  | | � S(   s9    Each instance of :class:'Bottle' is a WSGI application. (   R�  (   RI   R�   R6  (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    (   s   before_requests   after_requests	   app_resets   configN(%   RK   RL   Rp   R�   Rc   R_   R  R&  R(  Rr   R'  R*  R,  R  R/  RE  RG  R   RL  Rg   R�   RJ   RN  R�   RV  RF  RA  R�   RZ  R\  R^  R�   Re  Ri  Rs  R�  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  >  s@   					0	
							)		#H	t   BaseRequestc           B�  s;  e  Z d  Z d Z d Z d@ d � Z e d d d e �d �  � Z	 e d d d e �d �  � Z
 e d d	 d e �d
 �  � Z e d �  � Z e d �  � Z e d d d e �d �  � Z d@ d � Z e d d d e �d �  � Z d@ d@ d � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z e d d d e �d �  � Z d �  Z d �  Z e d d d e �d  �  � Z d! �  Z e d" �  � Z e d# �  � Z e Z e d d$ d e �d% �  � Z e d& �  � Z  e d d' d e �d( �  � Z! e d) �  � Z" e d* �  � Z# e d+ �  � Z$ d, d- � Z% e d. �  � Z& e d/ �  � Z' e d0 �  � Z( e d1 �  � Z) e d2 �  � Z* e d3 �  � Z+ e d4 �  � Z, d5 �  Z- d@ d6 � Z. d7 �  Z/ d8 �  Z0 d9 �  Z1 d: �  Z2 d; �  Z3 d< �  Z4 d= �  Z5 d> �  Z6 d? �  Z7 RS(A   sd   A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    R�   i � c         C�  s,   | d k r i  n | |  _ |  |  j d <d S(   s!    Wrap a WSGI environ dictionary. s   bottle.requestN(   Rg   R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    s
   bottle.appRb   c         C�  s   t  d � � d S(   s+    Bottle application handling this request. s0   This request is not connected to an application.N(   t   RuntimeError(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    s   bottle.routec         C�  s   t  d � � d S(   s=    The bottle :class:`Route` object that matches this request. s)   This request is not connected to a route.N(   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRA  �  s    s   route.url_argsc         C�  s   t  d � � d S(   s'    The arguments extracted from the URL. s)   This request is not connected to a route.N(   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s    d |  j  j d d � j d � S(   s�    The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). R�   R�   R�   (   R�   R�   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    c         C�  s   |  j  j d d � j �  S(   s6    The ``REQUEST_METHOD`` value as an uppercase string. R�   R�   (   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    s   bottle.request.headersc         C�  s   t  |  j � S(   sf    A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. (   t   WSGIHeaderDictR�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j | | � S(   sA    Return the value of a request header, or a given default value. (   R�  R�   (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_header  s    s   bottle.request.cookiesc         C�  s5   t  |  j j d d � � j �  } t d �  | D� � S(   s�    Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. t   HTTP_COOKIER�   c         s�  s!   |  ] } | j  | j f Vq d  S(   N(   Ra   Rm   (   R�   t   c(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R*   R�   R�   t   valuest	   FormsDict(   RI   t   cookies(    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    !c         C�  sY   |  j  j | � } | rO | rO t | | � } | rK | d | k rK | d S| S| pX | S(   s   Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. i    i   (   R�  R�   t   cookie_decode(   RI   Ra   R
   t   secretRm   t   dec(    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_cookie  s
    "s   bottle.request.queryc         C�  sT   t  �  } |  j d <t |  j j d d � � } x | D] \ } } | | | <q6 W| S(   s    The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. s
   bottle.gett   QUERY_STRINGR�   (   R�  R�   t
   _parse_qslR�   (   RI   R�   t   pairsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   )  s
    s   bottle.request.formsc         C�  sI   t  �  } x9 |  j j �  D]( \ } } t | t � s | | | <q q W| S(   s   Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. (   R�  RY  t   allitemsR>   t
   FileUpload(   RI   t   formsR�   t   item(    (    s&   /home/lgardner/git/professor/bottle.pyR�  5  s
    	s   bottle.request.paramsc         C�  sa   t  �  } x' |  j j �  D] \ } } | | | <q Wx' |  j j �  D] \ } } | | | <qC W| S(   s�    A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. (   R�  R�   R�  R�  (   RI   t   paramsRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  A  s    	s   bottle.request.filesc         C�  sI   t  �  } x9 |  j j �  D]( \ } } t | t � r | | | <q q W| S(   s�    File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.

        (   R�  RY  R�  R>   R�  (   RI   t   filesR�   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  L  s
    	s   bottle.request.jsonc         C�  sX   |  j  j d d � j �  j d � d } | d k rT |  j �  } | sJ d St | � Sd S(   s�    If the ``Content-Type`` header is ``application/json``, this
            property holds the parsed content of the request body. Only requests
            smaller than :attr:`MEMFILE_MAX` are processed to avoid memory
            exhaustion. t   CONTENT_TYPER�   t   ;i    s   application/jsonN(   R�   R�   t   lowerR@  t   _get_body_stringRg   t
   json_loads(   RI   t   ctypet   b(    (    s&   /home/lgardner/git/professor/bottle.pyt   jsonX  s    (
c         c�  sW   t  d |  j � } x> | rR | t | | � � } | s: Pn  | V| t | � 8} q Wd  S(   Ni    (   t   maxt   content_lengtht   minR}   (   RI   Ro  t   bufsizet   maxreadt   part(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _iter_bodyf  s    	 c         c�  s�  t  d d � } t d � t d � t d � } } } xYt r�| d � } xT | d | k r� | d � } | | 7} | s� | � n  t | � | k rM | � qM qM W| j | � \ }	 }
 }
 y t t |	 j �  � d � } Wn t k
 r� | � n X| d	 k rPn  | } xg | d	 k rq| s5| t	 | | � � } n  | |  | | } } | sY| � n  | V| t | � 8} qW| d
 � | k r8 | � q8 q8 Wd  S(   Ni�  s*   Error while parsing chunked transfer body.s   
R�  R�   i   i����i   i    i   (
   R�   RC   R�   R}   t	   partitionR�   t   tonatRP  R�   R�  (   RI   Ro  R�  RF   t   rnt   semt   bst   headerR�  t   sizeR�   R�  t   buffR�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _iter_chunkedn  s:    &	
 	 
  	s   bottle.request.bodyc         C�  s�   |  j  r |  j n |  j } |  j d j } t �  d t } } } x� | | |  j � D]n } | j | � | t	 | � 7} | rU | |  j k rU t
 d d � | } } | j | j �  � ~ t } qU qU W| |  j d <| j d � | S(   Ns
   wsgi.inputi    R�   s   w+b(   t   chunkedR�  R�  R�   Ro  R,   Rq   t   MEMFILE_MAXR   R}   R   t   getvalueR�   t   seek(   RI   t	   body_itert	   read_funcR3  t	   body_sizet   is_temp_fileR�  t   tmp(    (    s&   /home/lgardner/git/professor/bottle.pyt   _body�  s    c         C�  s�   |  j  } | |  j k r* t d d � � n  | d k  rF |  j d } n  |  j j | � } t | � |  j k r t d d � � n  | S(   s~    read body until content-length or MEMFILE_MAX into a string. Raise
            HTTPError(413) on requests that are to large. i�  s   Request to largei    i   (   R�  R�  R�   R3  Ro  R}   (   RI   t   clenR   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	 c         C�  s   |  j  j d � |  j  S(   sl   The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. i    (   R�  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR3  �  s    c         C�  s   d |  j  j d d � j �  k S(   s(    True if Chunked transfer encoding was. R�  t   HTTP_TRANSFER_ENCODINGR�   (   R�   R�   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    s   bottle.request.postc   	      C�  sw  t  �  } |  j j d � s[ t t |  j �  d � � } x | D] \ } } | | | <q= W| Si d d 6} x1 d D]) } | |  j k ro |  j | | | <qo qo Wt d |  j d	 | d
 t	 � } t
 r� t | d d d d d �| d <n t r� d | d <n  t j | �  } | |  d <| j pg  } xR | D]J } | j r_t | j | j | j | j � | | j <q%| j | | j <q%W| S(   s�    The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        s
   multipart/R)   R�   R�  R�   R�  t   CONTENT_LENGTHt   fpR�   t   keep_blank_valuesR(   R=   t   newlines   
s   _cgi.FieldStorage(   s   REQUEST_METHODs   CONTENT_TYPER�  (   R�  t   content_typet
   startswithR�  R�  R�  R�   R]   R3  R�   t   py31RH   R  t   cgit   FieldStorageR[   t   filenameR�  t   fileR�   R�  Rm   (	   RI   RZ  R�  Ra   Rm   t   safe_envR�   R   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRY  �  s2    	 
	c         C�  s   |  j  j �  S(   s�    The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. (   t   urlpartst   geturl(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    s   bottle.request.urlpartsc         C�  s�   |  j  } | j d � p' | j d d � } | j d � pE | j d � } | s� | j d d � } | j d � } | r� | | d k r� d	 n d
 k r� | d | 7} q� n  t |  j � } t | | | | j d � d � S(   s�    The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. t   HTTP_X_FORWARDED_PROTOs   wsgi.url_schemet   httpt   HTTP_X_FORWARDED_HOSTt	   HTTP_HOSTt   SERVER_NAMEs	   127.0.0.1t   SERVER_PORTt   80t   443t   :R�  R�   (   R�   R�   t   urlquotet   fullpatht   UrlSplitResult(   RI   t   envR�  t   hostt   portR�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	!$c         C�  s   t  |  j |  j j d � � S(   s:    Request path including :attr:`script_name` (if present). R�   (   R#   t   script_nameR�   RQ  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  j d d � S(   sh    The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. R�  R�   (   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   query_string�  s    c         C�  s4   |  j  j d d � j d � } | r0 d | d Sd S(   s�    The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. RO  R�   R�   (   R�   R�   RP  (   RI   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    i   c         C�  s<   |  j  j d d � } t | |  j | � \ |  d <|  d <d S(   s�    Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.

           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        RO  R�   R�   N(   R�   R�   R8  R�   (   RI   t   shiftt   script(    (    s&   /home/lgardner/git/professor/bottle.pyR8  	  s    c         C�  s   t  |  j j d � p d � S(   s�    The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. R�  i����(   R�   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j d d � j �  S(   sA    The Content-Type header as a lowercase-string (default: empty). R�  R�   (   R�   R�   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s%   |  j  j d d � } | j �  d k S(   s�    True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). t   HTTP_X_REQUESTED_WITHR�   t   xmlhttprequest(   R�   R�   R�  (   RI   t   requested_with(    (    s&   /home/lgardner/git/professor/bottle.pyt   is_xhr  s    c         C�  s   |  j  S(   s9    Alias for :attr:`is_xhr`. "Ajax" is not the right term. (   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   is_ajax'  s    c         C�  sK   t  |  j j d d � � } | r% | S|  j j d � } | rG | d f Sd S(   s�   HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. t   HTTP_AUTHORIZATIONR�   t   REMOTE_USERN(   t
   parse_authR�   R�   Rg   (   RI   t   basict   ruser(    (    s&   /home/lgardner/git/professor/bottle.pyt   auth,  s      
c         C�  sa   |  j  j d � } | r> g  | j d � D] } | j �  ^ q( S|  j  j d � } | r] | g Sg  S(   s(   A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. t   HTTP_X_FORWARDED_FORR�   t   REMOTE_ADDR(   R�   R�   R@  RP  (   RI   t   proxyt   ipt   remote(    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_route:  s
     &c         C�  s   |  j  } | r | d Sd S(   sg    The client IP as a string. Note that this information can be forged
            by malicious clients. i    N(   R�  Rg   (   RI   RA  (    (    s&   /home/lgardner/git/professor/bottle.pyt   remote_addrE  s    	c         C�  s   t  |  j j �  � S(   sD    Return a new :class:`Request` with a shallow :attr:`environ` copy. (   t   RequestR�   t   copy(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  L  s    c         C�  s   |  j  j | | � S(   N(   R�   R�   (   RI   Rm   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   P  s    c         C�  s   |  j  | S(   N(   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __getitem__Q  s    c         C�  s   d |  | <|  j  | =d  S(   NR�   (   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delitem__R  s   
 c         C�  s   t  |  j � S(   N(   Ru  R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  S  s    c         C�  s   t  |  j � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __len__T  s    c         C�  s   |  j  j �  S(   N(   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   U  s    c         C�  s�   |  j  j d � r! t d � � n  | |  j  | <d } | d k rI d } n- | d
 k r^ d } n | j d � rv d } n  x% | D] } |  j  j d | d � q} Wd S(   sA    Change an environ value and clear all caches that depend on it. s   bottle.request.readonlys$   The environ dictionary is read-only.s
   wsgi.inputR3  R�  R�  R�  RZ  R�  R�  R�   t   HTTP_R�  R�  s   bottle.request.N(    (   s   bodys   formss   filess   paramss   posts   json(   s   querys   params(   s   headerss   cookies(   R�   R�   R�   R�  R�   Rg   (   RI   Ra   Rm   t   todelete(    (    s&   /home/lgardner/git/professor/bottle.pyt   __setitem__V  s    			c         C�  s   d |  j  j |  j |  j f S(   Ns   <%s: %s %s>(   t	   __class__RK   R�   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  i  s    c         C�  s]   y5 |  j  d | } t | d � r0 | j |  � S| SWn! t k
 rX t d | � � n Xd S(   s@    Search in self.environ for additional user defined attributes. s   bottle.request.ext.%sRl   s   Attribute %r not defined.N(   R�   R2   Rl   R�   RO   (   RI   R�   t   var(    (    s&   /home/lgardner/git/professor/bottle.pyt   __getattr__l  s
    $c         C�  s4   | d k r t  j |  | | � S| |  j d | <d  S(   NR�   s   bottle.request.ext.%s(   t   objectt   __setattr__R�   (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  t  s     N(8   RK   RL   Rp   t	   __slots__R�  Rg   Rc   R_   R�   R�   RA  R�   R  R�   R�   R�  R�  R�  R�  R�   R�  R�  R�  R�  R�  R�  R�  R�  R3  R�  R�   RY  R�   R�  R�  R�  R�  R8  R�  R�  R�  R�  R�  R�  R�  R�  R�   R�  R�  Rp  R�  R�   R�  R  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  sd   			
#	
									c         C�  s   |  j  �  j d d � S(   NR�   t   -(   t   titlet   replace(   R0   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _hkey{  s    t   HeaderPropertyc           B�  s5   e  Z d e d  d � Z d �  Z d �  Z d �  Z RS(   R�   c         C�  s=   | | |  _  |  _ | | |  _ |  _ d | j �  |  _ d  S(   Ns   Current value of the %r header.(   R�   R
   t   readert   writerR�  Rp   (   RI   R�   R  R  R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  sE   | d  k r |  S| j j |  j |  j � } |  j rA |  j | � S| S(   N(   Rg   R�  R�   R�   R
   R  (   RI   Ri   Rj   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRl   �  s     c         C�  s   |  j  | � | j |  j <d  S(   N(   R  R�  R�   (   RI   Ri   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRn   �  s    c         C�  s   | j  |  j =d  S(   N(   R�  R�   (   RI   Ri   (    (    s&   /home/lgardner/git/professor/bottle.pyRo   �  s    N(   RK   RL   Rg   R�   Rc   Rl   Rn   Ro   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR    s   		t   BaseResponsec        
   B�  s�  e  Z d  Z d Z d Z i e d+ � d 6e d, � d 6Z d d- d- d � Z d- d � Z	 d �  Z
 d �  Z e d �  � Z e d �  � Z d �  Z d �  Z e e e d- d � Z [ [ e d �  � Z d �  Z d �  Z d �  Z d �  Z d- d � Z d �  Z d �  Z d �  Z e d  �  � Z e d � Z e d d! e �Z e d" d! d# �  d$ d% �  �Z  e d& d' � � Z! d- d( � Z" d) �  Z# d* �  Z$ RS(.   s�   Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    i�   s   text/html; charset=UTF-8s   Content-Typei�   R�   s   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Md5s   Last-Modifiedi0  R�   c         K�  s�   d  |  _ i  |  _ | |  _ | p' |  j |  _ | r{ t | t � rQ | j �  } n  x' | D] \ } } |  j	 | | � qX Wn  | r� x- | j �  D] \ } } |  j	 | | � q� Wn  d  S(   N(
   Rg   t   _cookiest   _headersR3  t   default_statusR1  R>   R]   t   itemsR2  (   RI   R3  R1  R�  t   more_headersR�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    			c         C�  s�   | p	 t  } t | t  � s! t � | �  } |  j | _ t d �  |  j j �  D� � | _ |  j r� t �  | _ | j j	 |  j j
 d d � � n  | S(   s    Returns a copy of self. c         s�  s"   |  ] \ } } | | f Vq d  S(   N(    (   R�   t   kt   v(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>�  s    R�  R�   (   R  t
   issubclasst   AssertionErrorR1  R]   R  R	  R  R*   RW  t   output(   RI   Rj   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	"	"c         C�  s   t  |  j � S(   N(   Ru  R3  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    c         C�  s&   t  |  j d � r" |  j j �  n  d  S(   NRJ   (   R2   R3  RJ   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s    c         C�  s   |  j  S(   s;    The HTTP status line as a string (e.g. ``404 Not Found``).(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   status_line�  s    c         C�  s   |  j  S(   s/    The HTTP status code as an integer (e.g. 404).(   R~  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRr  �  s    c         C�  s�   t  | t � r( | t j | � } } n= d | k rY | j �  } t | j �  d � } n t d � � d | k o| d k n s� t d � � n  | |  _ t | p� d | � |  _	 d  S(   Nt    i    s+   String status line without a reason phrase.id   i�  s   Status code out of range.s
   %d Unknown(
   R>   R�   t   _HTTP_STATUS_LINESR�   RP  R@  R�   R~  R�   R  (   RI   R1  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _set_status�  s     	c         C�  s   |  j  S(   N(   R  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _get_status�  s    sQ   A writeable property to change the HTTP response status. It accepts
            either a numeric code (100-999) or a string with a custom reason
            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and
            :data:`status_code` are updated accordingly. The return value is
            always a status string. c         C�  s   t  �  } |  j | _ | S(   sl    An instance of :class:`HeaderDict`, a case-insensitive dict-like
            view on the response headers. (   t
   HeaderDictR  R]   (   RI   t   hdict(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    	c         C�  s   t  | � |  j k S(   N(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __contains__�  s    c         C�  s   |  j  t | � =d  S(   N(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  t | � d S(   Ni����(   R  R  (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R  R  (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    |  j  j t | � | g � d S(   s|    Return the value of a previously defined header. If there is no
            header with that name, return a default value. i����(   R  R�   R  (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d S(   sh    Create a new response header, replacing any previously defined
            headers with the same name. N(   R�   R  R  (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_header   s    c         C�  s,   |  j  j t | � g  � j t | � � d S(   s=    Add an additional response header, not removing duplicates. N(   R  R�   R  R   R�   (   RI   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR2    s    c         C�  s   |  j  S(   sx    Yield (header, value) tuples, skipping headers that are not
            allowed with the current response status code. (   R4  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iter_headers	  s    c   	      C�  s  g  } t  |  j j �  � } d |  j k rF | j d |  j g f � n  |  j |  j k r� |  j |  j } g  | D] } | d | k ro | ^ qo } n  | g  | D]% \ } } | D] } | | f ^ q� q� 7} |  j r	x3 |  j j �  D] } | j d | j	 �  f � q� Wn  | S(   s.    WSGI conform list of (header, value) tuples. s   Content-Typei    s
   Set-Cookie(
   R[   R  R	  R   t   default_content_typeR~  t   bad_headersR  R�  t   OutputString(	   RI   Rx  R�  R  t   hR�   t   valst   valR�  (    (    s&   /home/lgardner/git/professor/bottle.pyR4    s    ,6	 R  t   Expiresc         C�  s   t  j t |  � � S(   N(   R   t   utcfromtimestampt
   parse_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   !  s    R  c         C�  s
   t  |  � S(   N(   t	   http_date(   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   "  s    s   UTF-8c         C�  s:   d |  j  k r6 |  j  j d � d j d � d j �  S| S(   sJ    Return the charset specified in the content-type header (default: utf8). s   charset=i����R�  i    (   R�  R@  RP  (   RI   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyRq  $  s    'c         K�  sk  |  j  s t �  |  _  n  | r< t t | | f | � � } n t | t � sZ t d � � n  t | � d k r{ t d � � n  | |  j  | <x� | j	 �  D]� \ } } | d k r� t | t
 � r� | j | j d d } q� n  | d k rFt | t t f � r
| j �  } n' t | t t f � r1t j | � } n  t j d | � } n  | |  j  | | j d	 d
 � <q� Wd S(   s�   Create a new cookie or replace an old one. If the `secret` parameter is
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
        s)   Secret key missing for non-string Cookie.i   s   Cookie value to long.t   max_agei   i  t   expiress   %a, %d %b %Y %H:%M:%S GMTR�   R�  N(   R  R*   R/   t   cookie_encodeR>   R?  RJ  R}   R�   R	  R   t   secondst   dayst   datedateR   t	   timetupleR�   R�   t   timet   gmtimet   strftimeR   (   RI   R�   Rm   R�  RC  Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   set_cookie+  s(    !	 c         K�  s+   d | d <d | d <|  j  | d | � d S(   sq    Delete a cookie. Be sure to use the same `domain` and `path`
            settings as used to create the cookie. i����R$  i    R%  R�   N(   R.  (   RI   Ra   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   delete_cookiec  s    

c         C�  sD   d } x7 |  j  D], \ } } | d | j �  | j �  f 7} q W| S(   NR�   s   %s: %s
(   R4  R�  RP  (   RI   Rx  R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  j  s    $(   s   Content-Type(   s   Allows   Content-Encodings   Content-Languages   Content-Lengths   Content-Ranges   Content-Types   Content-Md5s   Last-ModifiedN(%   RK   RL   Rp   R  R  R\   R  Rg   Rc   R�  Rp  RJ   R  R  Rr  R  R  R1  R�  R  R�  R�  R�  R�  R  R2  R  R4  R  R�  R�   R�  R%  Rq  R.  R/  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  sN    														8	c         �  s_   |  r t  d � n  t j �  �  �  f d �  } �  f d �  } �  f d �  } t | | | d � S(   Ns3   local_property() is deprecated and will be removed.c         �  s/   y �  j  SWn t k
 r* t d � � n Xd  S(   Ns    Request context not initialized.(   R�  RO   R�  (   RI   (   t   ls(    s&   /home/lgardner/git/professor/bottle.pyt   fgett  s     c         �  s   | �  _  d  S(   N(   R�  (   RI   Rm   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fsetx  s    c         �  s
   �  `  d  S(   N(   R�  (   RI   (   R0  (    s&   /home/lgardner/git/professor/bottle.pyt   fdely  s    s   Thread-local property(   RY   t	   threadingt   localR  (   R�   R1  R2  R3  (    (   R0  s&   /home/lgardner/git/professor/bottle.pyt   local_propertyq  s     t   LocalRequestc           B�  s    e  Z d  Z e j Z e �  Z RS(   sT   A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). (   RK   RL   Rp   R�  Rc   Rg  R6  R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR7  }  s   	t   LocalResponsec           B�  sD   e  Z d  Z e j Z e �  Z e �  Z e �  Z	 e �  Z
 e �  Z RS(   s+   A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    (   RK   RL   Rp   R  Rc   Rg  R6  R  R~  R  R  R3  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  �  s   					R9  c           B�  s#   e  Z d  d d d � Z d �  Z RS(   R�   c         K�  s#   t  t |  � j | | | | � d  S(   N(   t   superR9  Rc   (   RI   R3  R1  R�  R
  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  s@   |  j  | _  |  j | _ |  j | _ |  j | _ |  j | _ d  S(   N(   R~  R  R  R  R3  (   RI   Rh  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s
    N(   RK   RL   Rg   Rc   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR9  �  s   R�   c           B�  s#   e  Z d  Z d d d d d � Z RS(   i�  c         K�  s2   | |  _  | |  _ t t |  � j | | | � d  S(   N(   t	   exceptiont	   tracebackR9  R�   Rc   (   RI   R1  R3  R:  R;  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    		N(   RK   RL   R  Rg   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s   t   PluginErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR<  �  s    R!  c           B�  s)   e  Z d  Z d Z e d � Z d �  Z RS(   R�  i   c         C�  s   | |  _  d  S(   N(   R   (   RI   R   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         �  s)   |  j  � � s �  S�  � f d �  } | S(   Nc          �  s�   y �  |  | �  } Wn t  k
 r/ t �  } n Xt | t � rX � | � } d t _ | St | t � r� t | j t � r� � | j � | _ d | _ n  | S(   Ns   application/json(   R�   R   R>   R]   Rh  R�  R9  R3  (   R4   RR   t   rvt   json_response(   R�   R   (    s&   /home/lgardner/git/professor/bottle.pyRP   �  s    	!(   R   (   RI   R�   RA  RP   (    (   R�   R   s&   /home/lgardner/git/professor/bottle.pyR�   �  s
    	 (   RK   RL   R�   R  R   Rc   R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR!  �  s   R"  c           B�  s#   e  Z d  Z d Z d Z d �  Z RS(   s   This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. Rb  i   c         C�  s{   | j  j d � } t | t t f � rT t | � d k rT t | d | d � | � St | t � rs t | � | � S| Sd  S(   NRb  i   i    i   (   R�   R�   R>   RZ   R[   R}   t   viewR�   (   RI   R�   RA  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    '(   RK   RL   Rp   R�   R  R�   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR"  �  s   t   _ImportRedirectc           B�  s&   e  Z d  �  Z d d � Z d �  Z RS(   c         C�  sv   | |  _  | |  _ t j j | t j | � � |  _ |  j j j	 i t
 d 6g  d 6g  d 6|  d 6� t j j |  � d S(   s@    Create a virtual package that redirects imports (see PEP 302). t   __file__t   __path__t   __all__t
   __loader__N(   R�   t   impmaskR   t   modulesR�   t   impt
   new_modulet   moduleRs   t   updateRA  t	   meta_pathR   (   RI   R�   RE  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    		!c         C�  s=   d | k r d  S| j  d d � d } | |  j k r9 d  S|  S(   Nt   .i   i    (   t   rsplitR�   (   RI   t   fullnameR�   t   packname(    (    s&   /home/lgardner/git/professor/bottle.pyt   find_module�  s      c         C�  s   | t  j k r t  j | S| j d d � d } |  j | } t | � t  j | } t  j | <t |  j | | � |  | _ | S(   NRL  i   (   R   RF  RM  RE  t
   __import__Ru   RI  RD  (   RI   RN  t   modnamet   realnameRI  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_module�  s     
	N(   RK   RL   Rc   Rg   RP  RT  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR@  �  s   		t	   MultiDictc           B�  s
  e  Z d  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 e r� d	 �  Z d
 �  Z d �  Z e
 Z e Z e Z e Z n? d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d d d d � Z d �  Z d �  Z d �  Z e Z e Z RS(   s�    This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    c         O�  s,   t  d �  t  | | �  j �  D� � |  _  d  S(   Nc         s�  s$   |  ] \ } } | | g f Vq d  S(   N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R	  (   RI   R4   R  (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s    c         C�  s   t  |  j � S(   N(   R}   R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   t  |  j � S(   N(   Ru  R]   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp    s    c         C�  s   | |  j  k S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR    s    c         C�  s   |  j  | =d  S(   N(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  | d S(   Ni����(   R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  | | � d  S(   N(   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   |  j  j �  S(   N(   R]   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�     s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s   |  ] } | d  Vq d S(   i����N(    (   R�   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>  s    (   R]   R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s%   |  ] \ } } | | d  f Vq d S(   i����N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>   s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR	     s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R�   R  t   vlR  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>"  s    (   R]   R	  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  !  s    c         C�  s$   g  |  j  j �  D] } | d ^ q S(   Ni����(   R]   R�  (   RI   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  )  s    c         C�  s0   g  |  j  j �  D] \ } } | | d f ^ q S(   Ni����(   R]   R	  (   RI   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  *  s    c         C�  s   |  j  j �  S(   N(   R]   t   iterkeys(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRW  +  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s   |  ] } | d  Vq d S(   i����N(    (   R�   R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>,  s    (   R]   t
   itervalues(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRX  ,  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s%   |  ] \ } } | | d  f Vq d S(   i����N(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>.  s    (   R]   t	   iteritems(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRY  -  s    c         C�  s   d �  |  j  j �  D� S(   Nc         s�  s.   |  ]$ \ } } | D] } | | f Vq q d  S(   N(    (   R�   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>0  s    (   R]   RY  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   iterallitems/  s    c         C�  s9   g  |  j  j �  D]% \ } } | D] } | | f ^ q  q S(   N(   R]   RY  (   RI   R  RV  R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  1  s    i����c         C�  sA   y) |  j  | | } | r$ | | � S| SWn t k
 r< n X| S(   s�   Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        (   R]   Rm  (   RI   Ra   R
   t   indexR�   R  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   4  s    
c         C�  s    |  j  j | g  � j | � d S(   s5    Add a new value to the list of values for this key. N(   R]   R�   R   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   E  s    c         C�  s   | g |  j  | <d S(   s1    Replace the list of values with a single value. N(   R]   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   I  s    c         C�  s   |  j  j | � p g  S(   s5    Return a (possibly empty) list of values for a key. (   R]   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   getallM  s    N(   RK   RL   Rp   Rc   R�  Rp  R  R�  R�  R�  R�   R  R�  R	  R�  RW  RX  RY  RZ  Rg   R�   R   R   R\  t   getonet   getlist(    (    (    s&   /home/lgardner/git/professor/bottle.pyRU    s<   																						R�  c           B�  sP   e  Z d  Z d Z e Z d d � Z d d � Z d d d � Z	 e
 �  d � Z RS(   s�   This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. R=   c         C�  sd   t  | t � r7 |  j r7 | j d � j | p3 |  j � St  | t � r\ | j | pX |  j � S| Sd  S(   NR)   (   R>   R?   t   recode_unicodeR@   RE   t   input_encodingRA   (   RI   R0   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _fixd  s
    c         C�  sq   t  �  } | p |  j } | _ t | _ xB |  j �  D]4 \ } } | j |  j | | � |  j | | � � q5 W| S(   s�    Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. (   R�  R`  Rq   R_  R�  R   Ra  (   RI   R(   R�  RB   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRE   l  s    		,c         C�  s7   y |  j  |  | | � SWn t t f k
 r2 | SXd S(   s7    Return the value as a unicode string, or the default. N(   Ra  Rf  R�   (   RI   R�   R
   R(   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   getunicodew  s    c         C�  sG   | j  d � r4 | j d � r4 t t |  � j | � S|  j | d | �S(   Nt   __R
   (   R�  RB  R9  R�  R�  Rb  (   RI   R�   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  ~  s    N(   RK   RL   Rp   R`  R�   R_  Rg   Ra  RE   Rb  R?   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  V  s   R  c           B�  sn   e  Z d  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 d d	 d
 � Z d �  Z RS(   sz    A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. c         O�  s,   i  |  _  | s | r( |  j | | �  n  d  S(   N(   R]   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	 c         C�  s   t  | � |  j k S(   N(   R  R]   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   |  j  t | � =d  S(   N(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  t | � d S(   Ni����(   R]   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s,   |  j  j t | � g  � j t | � � d  S(   N(   R]   R�   R  R   R�   (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s    c         C�  s    t  | � g |  j t | � <d  S(   N(   R�   R]   R  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR   �  s    c         C�  s   |  j  j t | � � p g  S(   N(   R]   R�   R  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR\  �  s    i����c         C�  s   t  j |  t | � | | � S(   N(   RU  R�   R  (   RI   Ra   R
   R[  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  sJ   xC g  | D] } t  | � ^ q
 D]" } | |  j k r  |  j | =q  q  Wd  S(   N(   R  R]   (   RI   t   namesR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   filter�  s    &N(   RK   RL   Rp   Rc   R  R�  R�  R�  R   R   R\  Rg   R�   Re  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s   								R�  c           B�  sq   e  Z d  Z d Z d �  Z d �  Z d d � Z d �  Z d �  Z	 d �  Z
 d	 �  Z d
 �  Z d �  Z d �  Z RS(   s    This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    R�  R�  c         C�  s   | |  _  d  S(   N(   R�   (   RI   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    c         C�  s3   | j  d d � j �  } | |  j k r+ | Sd | S(   s6    Translate header field name to CGI/WSGI environ key. R�  R�   R�  (   R   R�   t   cgikeys(   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   _ekey�  s    c         C�  s   |  j  j |  j | � | � S(   s:    Return the header value as is (may be bytes or unicode). (   R�   R�   Rg  (   RI   Ra   R
   (    (    s&   /home/lgardner/git/professor/bottle.pyt   raw�  s    c         C�  s   t  |  j |  j | � d � S(   NR)   (   R�  R�   Rg  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   t  d |  j � � d  S(   Ns   %s is read-only.(   RJ  R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   t  d |  j � � d  S(   Ns   %s is read-only.(   RJ  R�  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         c�  so   xh |  j  D]] } | d  d k r> | d j d d � j �  Vq
 | |  j k r
 | j d d � j �  Vq
 q
 Wd  S(   Ni   R�  R�   R�  (   R�   R   R�  Rf  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s
    c         C�  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   t  |  j �  � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  | � |  j k S(   N(   Rg  R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    (   s   CONTENT_TYPEs   CONTENT_LENGTHN(   RK   RL   Rp   Rf  Rc   Rg  Rg   Rh  R�  R�  R�  Rp  R�   R�  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   
								R�   c           B�  s�   e  Z d  Z d Z d e f d �  �  YZ d �  Z d �  Z d e d � Z	 d	 �  Z
 d
 �  Z d �  Z d �  Z d �  Z d d � Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z RS(   sH   A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.

        This storage is optimized for fast read access. Retrieving a key
        or using non-altering dict methods (e.g. `dict.get()`) has no overhead
        compared to a native dict.
    t   _metaR  t	   Namespacec           B�  s�   e  Z d  �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z	 d �  Z
 d	 �  Z d
 �  Z d �  Z d �  Z d �  Z RS(   c         C�  s   | |  _  | |  _ d  S(   N(   t   _configt   _prefix(   RI   R�   t	   namespace(    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	c         C�  s    t  d � |  j |  j d | S(   Ns}   Accessing namespaces as dicts is discouraged. Only use flat item access: cfg["names"]["pace"]["key"] -> cfg["name.space.key"]RL  (   RY   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
c         C�  s   | |  j  |  j d | <d  S(   NRL  (   Rk  Rl  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  |  j d | =d  S(   NRL  (   Rk  Rl  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         c�  sZ   |  j  d } xF |  j D]; } | j d � \ } } } | |  j  k r | r | Vq q Wd  S(   NRL  (   Rl  Rk  t
   rpartition(   RI   t	   ns_prefixRa   t   nst   dotR�   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s
    c         C�  s   g  |  D] } | ^ q S(   N(    (   RI   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         C�  s   t  |  j �  � S(   N(   R}   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    c         C�  s   |  j  d | |  j k S(   NRL  (   Rl  Rk  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s    c         C�  s   d |  j  S(   Ns   <Config.Namespace %s.*>(   Rl  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __str__�  s    c         C�  s�   t  d � | |  k rM | d j �  rM t j |  j |  j d | � |  | <n  | |  k rw | j d � rw t | � � n  |  j | � S(   Ns   Attribute access is deprecated.i    RL  Rc  (	   RY   t   isupperR�   Rj  Rk  Rl  R�  RO   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
'c         C�  s�   | d k r | |  j  | <d  St d � t t | � rE t d � � n  | |  k r� |  | r� t |  | |  j � r� t d � � n  | |  | <d  S(   NRk  Rl  s#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   s   _configs   _prefix(   Rs   RY   R2   R9   RO   R>   R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s    
,c         C�  so   | |  k rk |  j  | � } t | |  j � rk | d } x. |  D]# } | j | � r> |  | | =q> q> Wqk n  d  S(   NRL  (   R�   R>   R�  R�  (   RI   Ra   R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   __delattr__  s    
c         O�  s   t  d � |  j | | �  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1     s    
(   RK   RL   Rc   R�  R�  R�  Rp  R�   R�  R  R  Rr  R�  R�  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRj  �  s   														c         O�  sB   i  |  _  d �  |  _ | s! | r> t d � |  j | | �  n  d  S(   Nc         S�  s   d  S(   N(   Rg   (   R�   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR!     s    s-   Constructor does no longer accept parameters.(   Ri  R  RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyRc     s
    	
c         C�  sx   t  �  } | j | � x[ | j �  D]M } xD | j | � D]3 \ } } | d k rb | d | } n  | |  | <q9 Wq# W|  S(   s   Load values from an *.ini style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        t   DEFAULTt   bottleRL  (   Ru  s   bottle(   R-   Ro  t   sectionsR	  (   RI   R�  R�   t   sectionRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_config!  s    	R�   c   	      C�  s  | | f g } x� | r| j  �  \ } } t | t � sR t d t | � � � n  x� | j �  D]� \ } } t | t � s� t d t | � � � n  | r� | d | n | } t | t � r� | j | | f � | r� |  j |  | � |  | <q� q_ | |  | <q_ Wq W|  S(   s�    Import values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> ConfigDict().load_dict({'name': {'space': {'key': 'value'}}})
            {'name.space.key': 'value'}
        s   Source is not a dict (r)s   Key is not a string (%r)RL  (	   R�   R>   R]   RJ  R�   R	  R?  R   Rj  (	   RI   t   sourceRm  R�   t   stackR�   Ra   Rm   t   full_key(    (    s&   /home/lgardner/git/professor/bottle.pyR�   1  s    	c         O�  s{   d } | rC t  | d t � rC | d j d � d } | d } n  x1 t | | �  j �  D] \ } } | |  | | <qY Wd S(   s�    If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` R�   i    RL  i   N(   R>   R?  RP  R]   R	  (   RI   R4   RR   R�   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ  I  s    "c         C�  s!   | |  k r | |  | <n  |  | S(   N(    (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   T  s    c         C�  s�   t  | t � s( t d t | � � � n  |  j | d d �  � | � } | |  k rf |  | | k rf d  S|  j | | � t j |  | | � d  S(   Ns   Key has type %r (not a string)Re  c         S�  s   |  S(   N(    (   R    (    (    s&   /home/lgardner/git/professor/bottle.pyR!   ]  s    (   R>   R?  RJ  R�   t   meta_getR  R]   R�  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  Y  s    c         C�  s   t  j |  | � d  S(   N(   R]   R�  (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  c  s    c         C�  s   x |  D] } |  | =q Wd  S(   N(    (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt   clearf  s    c         C�  s   |  j  j | i  � j | | � S(   s-    Return the value of a meta field for a key. (   Ri  R�   (   RI   Ra   t	   metafieldR
   (    (    s&   /home/lgardner/git/professor/bottle.pyR}  j  s    c         C�  s:   | |  j  j | i  � | <| |  k r6 |  | |  | <n  d S(   sq    Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. N(   Ri  R�   (   RI   Ra   R  Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR  n  s    c         C�  s   |  j  j | i  � j �  S(   s;    Return an iterable of meta field names defined for a key. (   Ri  R�   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   meta_listu  s    c         C�  sv   t  d � | |  k r? | d j �  r? |  j |  | � |  | <n  | |  k ri | j d � ri t | � � n  |  j | � S(   Ns   Attribute access is deprecated.i    Rc  (   RY   Rs  Rj  R�  RO   R�   (   RI   Ra   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  z  s    
c         C�  s�   | |  j  k r" t j |  | | � St d � t t | � rJ t d � � n  | |  k r� |  | r� t |  | |  j � r� t d � � n  | |  | <d  S(   Ns#   Attribute assignment is deprecated.s   Read-only attribute.s   Non-empty namespace attribute.(   R�  R]   R�  RY   R2   RO   R>   Rj  (   RI   Ra   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s    
,c         C�  so   | |  k rk |  j  | � } t | |  j � rk | d } x. |  D]# } | j | � r> |  | | =q> q> Wqk n  d  S(   NRL  (   R�   R>   Rj  R�  (   RI   Ra   R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRt  �  s    
c         O�  s   t  d � |  j | | �  |  S(   Ns8   Calling ConfDict is deprecated. Use the update() method.(   RY   RJ  (   RI   R4   RR   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    
(   s   _metas
   _on_changeN(   RK   RL   Rp   R�  R9   Rj  Rc   Ry  Rq   R�   RJ  R�   R�  R�  R~  Rg   R}  R  R�  R�  R�  Rt  R1   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s$   A					
						
		t   AppStackc           B�  s#   e  Z d  Z d �  Z d d � Z RS(   s>    A stack-like list. Calling it returns the head of the stack. c         C�  s   |  d S(   s)    Return the current default application. i����(    (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyR1   �  s    c         C�  s,   t  | t � s t �  } n  |  j | � | S(   s1    Add a new :class:`Bottle` instance to the stack (   R>   R  R   (   RI   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyt   push�  s    N(   RK   RL   Rp   R1   Rg   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	Rt  c           B�  s   e  Z d d � Z d �  Z RS(   i   i@   c         C�  sS   | | |  _  |  _ x9 d D]1 } t | | � r t |  | t | | � � q q Wd  S(   Nt   filenoRJ   Ro  t	   readlinest   tellR�  (   s   filenos   closes   reads	   readliness   tells   seek(   R�  t   buffer_sizeR2   Ru   Rh   (   RI   R�  R�  R`   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s     c         c�  s?   |  j  |  j } } x% t r: | | � } | s2 d  S| Vq Wd  S(   N(   R�  Ro  R�   (   RI   R�  Ro  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    	 i   (   RK   RL   Rc   Rp  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRt  �  s   Rw  c           B�  s,   e  Z d  Z d d � Z d �  Z d �  Z RS(   s�    This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). c         C�  s   | |  _  t | � |  _ d  S(   N(   t   iteratorR^   t   close_callbacks(   RI   R�  RJ   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s    	c         C�  s   t  |  j � S(   N(   Ru  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    c         C�  s   x |  j  D] } | �  q
 Wd  S(   N(   R�  (   RI   Rf   (    (    s&   /home/lgardner/git/professor/bottle.pyRJ   �  s    N(   RK   RL   Rp   Rg   Rc   Rp  RJ   (    (    (    s&   /home/lgardner/git/professor/bottle.pyRw  �  s   	R  c           B�  sP   e  Z d  Z d e d d � Z d	 d	 e d � Z d �  Z d �  Z	 d d � Z RS(
   sf   This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    s   ./t   allc         C�  s1   t  |  _ | |  _ | |  _ g  |  _ i  |  _ d  S(   N(   t   opent   openert   baset	   cachemodeR�   t   cache(   RI   R�  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �  s
    				c         C�  s�   t  j j t  j j | p |  j � � } t  j j t  j j | t  j j | � � � } | t  j 7} | |  j k r� |  j j | � n  | r� t  j j | � r� t  j	 | � n  | d k r� |  j j | � n |  j j | | � |  j j �  t  j j | � S(   s   Add a new path to the list of search paths. Return False if the
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
        N(   t   osR�   t   abspatht   dirnameR�  R�   t   sepR+  t   isdirt   makedirsRg   R   R)  R�  R~  t   exists(   RI   R�   R�  R[  t   create(    (    s&   /home/lgardner/git/professor/bottle.pyt   add_path�  s    '-c         c�  s�   |  j  } x� | r� | j �  } t j  j | � s7 q n  xS t j | � D]B } t j  j | | � } t j  j | � r� | j | � qG | VqG Wq Wd S(   s:    Iterate over all existing files in all registered paths. N(   R�   R�   R�  R�  t   listdirR�   R   (   RI   t   searchR�   R�   t   full(    (    s&   /home/lgardner/git/professor/bottle.pyRp  �  s    
	  c         C�  s�   | |  j  k s t r� x[ |  j D]P } t j j | | � } t j j | � r |  j d k rk | |  j  | <n  | Sq W|  j d k r� d |  j  | <q� n  |  j  | S(   s�    Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. R�  t   found(   s   alls   foundN(   R�  R�   R�   R�  R�   t   isfileR�  Rg   (   RI   R�   R�   t   fpath(    (    s&   /home/lgardner/git/professor/bottle.pyt   lookup	  s    t   rc         O�  sA   |  j  | � } | s( t d | � � n  |  j | d | | | �S(   s=    Find a resource and return a file object, or raise IOError. s   Resource %r not found.R�   (   R�  t   IOErrorR�  (   RI   R�   R�   R�   R.  t   fname(    (    s&   /home/lgardner/git/professor/bottle.pyR�  	  s     N(
   RK   RL   Rp   R�  Rc   Rg   Rq   R�  Rp  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s   
		R�  c           B�  sb   e  Z d d  � Z e d � Z e d d e d d �Z e d �  � Z	 d d	 � Z
 e d d
 � Z RS(   c         C�  s=   | |  _  | |  _ | |  _ | r- t | � n t �  |  _ d S(   s    Wrapper for file uploads. N(   R�  R�   t   raw_filenameR  R�  (   RI   t   fileobjR�   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   "	  s    			s   Content-Types   Content-LengthR  R
   i����c         C�  s�   |  j  } t | t � s- | j d d � } n  t d | � j d d � j d � } t j j | j	 d t j j
 � � } t j d d | � j �  } t j d d	 | � j d
 � } | d  p� d S(   s�   Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        R=   t   ignoret   NFKDt   ASCIIs   \s   [^a-zA-Z0-9-_.\s]R�   s   [-\s]+R�  s   .-i�   t   empty(   R�  R>   R?   RE   R   R@   R�  R�   t   basenameR   R�  R�   R�   RP  (   RI   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  0	  s    
	$$i   i   c         C�  sa   |  j  j | j |  j  j �  } } } x$ | | � } | s? Pn  | | � q) W|  j  j | � d  S(   N(   R�  Ro  R   R�  R�  (   RI   R�  t
   chunk_sizeRo  R   R�   t   buf(    (    s&   /home/lgardner/git/professor/bottle.pyt
   _copy_fileC	  s    & c         C�  s�   t  | t � r� t j j | � r< t j j | |  j � } n  | rd t j j | � rd t d � � n  t	 | d � � } |  j
 | | � Wd QXn |  j
 | | � d S(   s�   Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        s   File exists.t   wbN(   R>   R?  R�  R�   R�  R�   R�  R�  R�  R�  R�  (   RI   t   destinationt	   overwriteR�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   saveK	  s    	Ni   i   (   RK   RL   Rg   Rc   R  R�  R�   R�  Rr   R�  R�  Rq   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�   	  s   i�  s   Unknown Error.c         C�  s   t  |  | � � d S(   s+    Aborts execution and causes a HTTP error. N(   R�   (   R`  t   text(    (    s&   /home/lgardner/git/professor/bottle.pyt   aborth	  s    c         C�  st   | s* t  j d � d k r! d n d } n  t j d t � } | | _ d | _ | j d t t  j	 |  � � | � d S(	   sd    Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. t   SERVER_PROTOCOLs   HTTP/1.1i/  i.  Rj   R�   t   LocationN(
   R7  R�   Rh  R�  R9  R1  R3  R  R#   R�   (   R�   R`  Rd  (    (    s&   /home/lgardner/git/professor/bottle.pyt   redirectm	  s    $		i   c         c�  s[   |  j  | � xG | d k rV |  j t | | � � } | s> Pn  | t | � 8} | Vq Wd S(   sF    Yield chunks from a range in a file. No chunk is bigger than maxread.i    N(   R�  Ro  R�  R}   (   R�  R�   RA   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _file_iter_rangey	  s     t   autos   UTF-8c         C�  s@  t  j j | � t  j } t  j j t  j j | |  j d � � � }  t �  } |  j | � sh t d d � St  j j	 |  � s� t  j j
 |  � r� t d d � St  j |  t  j � s� t d d � S| d k r� t j |  � \ } } | r� | | d <q� n  | r:| d	  d
 k r-| r-d | k r-| d | 7} n  | | d <n  | rut  j j | t k r[|  n | � } d | | d <n  t  j |  � } | j | d <} t j d t j | j � � }	 |	 | d <t j j d � }
 |
 r�t |
 j d � d j �  � }
 n  |
 d% k	 rD|
 t | j � k rDt j d t j �  � | d <t d d | � St j d k rYd n t  |  d � } d | d <t j j d � } d t j k r3t! t" t j d | � � } | s�t d d  � S| d \ } } d! | | d" | f | d# <t# | | � | d <| r t$ | | | | � } n  t | d d$ | �St | | � S(&   s�   Open a file in a safe way and return :exc:`HTTPResponse` with status
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
    s   /\i�  s   Access denied.i�  s   File does not exist.s/   You do not have permission to access this file.R�  s   Content-Encodingi   s   text/Rq  s   ; charset=%ss   Content-Types   attachment; filename="%s"s   Content-Dispositions   Content-Lengths   %a, %d %b %Y %H:%M:%S GMTs   Last-Modifiedt   HTTP_IF_MODIFIED_SINCER�  i    t   DateR1  i0  R�   R�   t   rbRA   s   Accept-Rangest
   HTTP_RANGEi�  s   Requested Range Not Satisfiables   bytes %d-%d/%di   s   Content-Rangei�   N(%   R�  R�   R�  R�  R�   RP  R]   R�  R�   R�  R�  t   accesst   R_OKt	   mimetypest
   guess_typeR�  R�   t   statt   st_sizeR+  R-  R,  t   st_mtimeR7  R�   R�   R"  R@  Rg   R�   R9  R�   R�  R[   t   parse_range_headerR�   R�  (   R�  t   roott   mimetypet   downloadRq  R�  R(   t   statsR�  t   lmt   imsR3  t   rangesR�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   static_file�	  sX    *	& "$
"!$
 c         C�  s&   |  r t  j d � n  t |  � a d S(   sS    Change the debug level.
    There is only one debug level supported at the moment.R
   N(   RT   t   simplefilterR  R�   (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   debug�	  s     c         C�  ss   t  |  t t f � r$ |  j �  }  n' t  |  t t f � rK t j |  � }  n  t  |  t � so t j	 d |  � }  n  |  S(   Ns   %a, %d %b %Y %H:%M:%S GMT(
   R>   R)  R   t   utctimetupleR�   R�   R+  R,  R?  R-  (   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR#  �	  s    c         C�  se   y@ t  j j |  � } t j | d  d � | d p6 d t j SWn t t t t	 f k
 r` d SXd S(   sD    Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. i   i    i	   N(   i    (   t   emailt   utilst   parsedate_tzR+  t   mktimet   timezoneRJ  R�   t
   IndexErrort   OverflowErrorRg   (   R�  t   ts(    (    s&   /home/lgardner/git/professor/bottle.pyR"  �	  s
    .c         C�  s�   ye |  j  d d � \ } } | j �  d k rd t t j t | � � � j  d d � \ } } | | f SWn t t f k
 r d SXd S(   s]    Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or Nonei   R�  R�  N(	   R@  Rg   R�  R/   t   base64t	   b64decodeRC   R�   R�   (   R�  R�   R   t   usert   pwd(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �	  s    -c         c�  s,  |  s |  d  d k r d Sg  |  d j  d � D]$ } d | k r/ | j  d d � ^ q/ } x� | D]� \ } } y� | s� t d | t | � � | } } nB | s� t | � | } } n& t | � t t | � d | � } } d | k o� | k  o� | k n r| | f Vn  Wq` t k
 r#q` Xq` Wd S(   s~    Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive.i   s   bytes=NR�   R�  i   i    (   R@  R�  R�   R�  R�   (   R�  t   maxlenR�  R�  R�   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �	  s     >#&'c         C�  s�   g  } x� |  j  d d � j d � D]� } | s4 q" n  | j d d � } t | � d k rh | j d � n  t | d j  d d	 � � } t | d j  d d	 � � } | j | | f � q" W| S(
   NR�  t   &t   =i   i   R�   i    t   +R  (   R   R@  R}   R   t
   urlunquote(   t   qsR�  t   pairt   nvRa   Rm   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  
  s    "  c         C�  s6   t  d �  t |  | � D� � o5 t |  � t | � k S(   ss    Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. c         s�  s-   |  ]# \ } } | | k r! d  n d Vq d S(   i    i   N(    (   R�   R    t   y(    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>
  s    (   t   sumt   zipR}   (   R4   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _lscmp
  s    c         C�  s^   t  j t j |  d � � } t  j t j t | � | � j �  � } t d � | t d � | S(   s>    Encode and sign a pickle-able object. Return a (byte) string i����t   !R�   (   R�  t	   b64encodet   pickleR   t   hmact   newRC   t   digest(   R   Ra   R�   t   sig(    (    s&   /home/lgardner/git/professor/bottle.pyR&  
  s    'c         C�  s�   t  |  � }  t |  � r� |  j t  d � d � \ } } t | d t j t j t  | � | � j �  � � r� t	 j
 t j | � � Sn  d S(   s?    Verify and decode an encoded string. Return an object or None.R�   i   N(   RC   t   cookie_is_encodedR@  R�  R�  R�  R�  R�  R�  R�  R   R�  Rg   (   R   Ra   R�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�   
  s    4c         C�  s+   t  |  j t d � � o' t d � |  k � S(   s9    Return True if the argument looks like a encoded cookie.R�  R�   (   R  R�  RC   (   R   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  *
  s    c         C�  s@   |  j  d d � j  d d � j  d d � j  d d � j  d	 d
 � S(   s;    Escape HTML special characters ``&<>`` and quotes ``'"``. R�  s   &amp;t   <s   &lt;t   >s   &gt;t   "s   &quot;t   's   &#039;(   R   (   t   string(    (    s&   /home/lgardner/git/professor/bottle.pyR�  /
  s    *c         C�  s2   d t  |  � j d d � j d d � j d d � S(   s;    Escape and quote a string to be used as an HTTP attribute.s   "%s"s   
s   &#10;s   s   &#13;s   	s   &#9;(   R�  R   (   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt
   html_quote5
  s    c         c�  s�   d |  j  j d d � j d � } t |  � } t | d � t | d pK g  � } | d | t | d |  � 7} | Vx) | d | D] } | d | 7} | Vq� Wd S(   s�   Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    R�   Rc  i    i   s   /<%s>N(   RK   R   RQ  R   R}   RZ   (   Rf   R�   t   spect   argct   arg(    (    s&   /home/lgardner/git/professor/bottle.pyRX  ;
  s    
"$ c   	      C�  s}  | d k r |  | f S| j  d � j d � } |  j  d � j d � } | re | d d k re g  } n  | r� | d d k r� g  } n  | d k r� | t | � k r� | |  } | | } | | } nh | d k  r| t | � k r| | } | | } | |  } n( | d k  rd n d } t d | � � d d j | � } d d j | � } | j d � rs| rs| d 7} n  | | f S(   sS   Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    i    R�   R�   RO  R�   s"   Cannot shift. Nothing left from %s(   RP  R@  R}   R  R�   RB  (	   R�  t	   path_infoR�  t   pathlistt
   scriptlistt   movedR�  t   new_script_namet   new_path_info(    (    s&   /home/lgardner/git/professor/bottle.pyR8  O
  s.    	 
 	 	



 t   privates   Access deniedc         �  s   �  � � f d �  } | S(   se    Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. c         �  s   � �  � � f d �  } | S(   Nc          �  se   t  j p d \ } } | d  k s1 �  | | � rX t d � � } | j d d � � | S� |  | �  S(   Ni�  s   WWW-Authenticates   Basic realm="%s"(   NN(   R7  R�  Rg   R�   R2  (   R4   RR   R�  t   passwordRF   (   t   checkRf   t   realmR�  (    s&   /home/lgardner/git/professor/bottle.pyRP   r
  s    (    (   Rf   RP   (   R�  R   R�  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  q
  s    (    (   R�  R   R�  R0  (    (   R�  R   R�  s&   /home/lgardner/git/professor/bottle.pyt
   auth_basicn
  s    	c         �  s+   t  j t t �  � � �  f d �  � } | S(   sA    Return a callable that relays calls to the current default app. c          �  s   t  t �  �  � |  | �  S(   N(   Rh   R�   (   R4   RR   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyRP   �
  s    (   RM   t   wrapsRh   R  (   R�   RP   (    (   R�   s&   /home/lgardner/git/professor/bottle.pyt   make_default_app_wrapper�
  s    'RA  R�   RZ  R\  R^  R�   RE  R/  R   RL  RV  t   ServerAdapterc           B�  s/   e  Z e Z d  d d � Z d �  Z d �  Z RS(   s	   127.0.0.1i�  c         K�  s%   | |  _  | |  _ t | � |  _ d  S(   N(   RC  R�  R�   R�  (   RI   R�  R�  RC  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   �
  s    		c         C�  s   d  S(   N(    (   RI   R_  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    c         C�  sU   d j  g  |  j j �  D]" \ } } d | t | � f ^ q � } d |  j j | f S(   Ns   , s   %s=%ss   %s(%s)(   R�   RC  R	  R�   R�  RK   (   RI   R  R  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s    A(   RK   RL   Rq   t   quietRc   RN  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   	t	   CGIServerc           B�  s   e  Z e Z d  �  Z RS(   c         �  s3   d d l  m } �  f d �  } | �  j | � d  S(   Ni����(   t
   CGIHandlerc         �  s   |  j  d d � �  |  | � S(   NR�   R�   (   R�   (   R�   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyt   fixed_environ�
  s    (   t   wsgiref.handlersR  RN  (   RI   R_  R  R  (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   R�   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   t   FlupFCGIServerc           B�  s   e  Z d  �  Z RS(   c         C�  sN   d d  l  } |  j j d |  j |  j f � | j j j | |  j � j �  d  S(   Ni����t   bindAddress(	   t   flup.server.fcgiRC  R�   R�  R�  t   servert   fcgit
   WSGIServerRN  (   RI   R_  t   flup(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR
  �
  s   t   WSGIRefServerc           B�  s   e  Z d  �  Z RS(   c         �  s�   d d l  m �  m } d d l  m } d d  l � d �  f �  � f d �  �  Y} � j j d | � } � j j d | � } d � j k r� t | d	 � � j	 k r� d
 | f � f d �  �  Y} q� n  | � j � j
 | | | � } | j �  d  S(   Ni����(   t   WSGIRequestHandlerR  (   t   make_servert   FixedHandlerc           �  s#   e  Z d  �  Z �  � f d �  Z RS(   c         S�  s   |  j  d S(   Ni    (   t   client_address(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   address_string�
  s    c          �  s   � j  s �  j |  | �  Sd  S(   N(   R  t   log_request(   R�   t   kw(   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s    	(   RK   RL   R  R  (    (   R  RI   (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   	t   handler_classt   server_classR�  t   address_familyt
   server_clsc           �  s   e  Z �  j Z RS(    (   RK   RL   t   AF_INET6R  (    (   t   socket(    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   (   t   wsgiref.simple_serverR  R  R  R  RC  R�   R�  Rh   t   AF_INETR�  t   serve_forever(   RI   R�   R  R  R  t   handler_clsR  t   srv(    (   R  RI   R  s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    "(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR  �
  s   t   CherryPyServerc           B�  s   e  Z d  �  Z RS(   c         C�  s�   d d l  m } |  j |  j f |  j d <| |  j d <|  j j d � } | r[ |  j d =n  |  j j d � } | r� |  j d =n  | j |  j �  } | r� | | _ n  | r� | | _ n  z | j	 �  Wd  | j
 �  Xd  S(   Ni����(   t
   wsgiservert	   bind_addrt   wsgi_appt   certfilet   keyfile(   t   cherrypyR%  R�  R�  RC  R�   t   CherryPyWSGIServert   ssl_certificatet   ssl_private_keyR�   t   stop(   RI   R_  R%  R(  R)  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s"    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR$  �
  s   t   WaitressServerc           B�  s   e  Z d  �  Z RS(   c         C�  s0   d d l  m } | | d |  j d |  j �d  S(   Ni����(   t   serveR�  R�  (   t   waitressR0  R�  R�  (   RI   R_  R0  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR/  �
  s   t   PasteServerc           B�  s   e  Z d  �  Z RS(   c         C�  se   d d l  m } d d l m } | | d |  j �} | j | d |  j d t |  j � |  j	 �d  S(   Ni����(   t
   httpserver(   t   TransLoggert   setup_console_handlerR�  R�  (
   t   pasteR3  t   paste.transloggerR4  R  R0  R�  R�   R�  RC  (   RI   R_  R3  R4  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �
  s
    !(   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR2  �
  s   t   MeinheldServerc           B�  s   e  Z d  �  Z RS(   c         C�  s:   d d l  m } | j |  j |  j f � | j | � d  S(   Ni����(   R  (   t   meinheldR  t   listenR�  R�  RN  (   RI   R_  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN     s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR8  �
  s   t   FapwsServerc           B�  s   e  Z d  Z d �  Z RS(   sA    Extremely fast webserver using libev. See http://www.fapws.org/ c         �  s�   d d  l  j } d d l m } m } |  j } t | j d � d k rV t | � } n  | j	 |  j
 | � d t j k r� |  j r� t d � t d � n  | j | � �  f d �  } | j d	 | f � | j �  d  S(
   Ni����(   R�  R�   i����g�������?t   BOTTLE_CHILDs3   WARNING: Auto-reloading does not work with Fapws3.
s/            (Fapws3 breaks python thread support)
c         �  s   t  |  d <�  |  | � S(   Ns   wsgi.multiprocess(   Rq   (   R�   R6  (   R_  (    s&   /home/lgardner/git/professor/bottle.pyR�     s    
R�   (   t   fapws._evwsgit   _evwsgit   fapwsR�  R�   R�  R�   t   SERVER_IDENTR�   R�   R�  R�  R�   R  t   _stderrt   set_base_modulet   wsgi_cbRN  (   RI   R_  t   evwsgiR�  R�   R�  R�   (    (   R_  s&   /home/lgardner/git/professor/bottle.pyRN    s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR;    s   t   TornadoServerc           B�  s   e  Z d  Z d �  Z RS(   s<    The super hyped asynchronous server by facebook. Untested. c         C�  s~   d d  l  } d d  l } d d  l } | j j | � } | j j | � } | j d |  j d |  j	 � | j
 j j �  j �  d  S(   Ni����R�  t   address(   t   tornado.wsgit   tornado.httpservert   tornado.ioloopR�  t   WSGIContainerR3  t
   HTTPServerR:  R�  R�  t   ioloopt   IOLoopt   instanceR�   (   RI   R_  t   tornadot	   containerR  (    (    s&   /home/lgardner/git/professor/bottle.pyRN    s
    $(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRE    s   t   AppEngineServerc           B�  s   e  Z d  Z e Z d �  Z RS(   s     Adapter for Google App Engine. c         �  sa   d d l  m � t j j d � } | rP t | d � rP �  � f d �  | _ n  � j �  � d  S(   Ni����(   t   utilR   t   mainc           �  s   � j  �  � S(   N(   t   run_wsgi_app(    (   R_  RR  (    s&   /home/lgardner/git/professor/bottle.pyR!   /  s    (   t   google.appengine.ext.webappRR  R   RF  R�   R2   RS  RT  (   RI   R_  RI  (    (   R_  RR  s&   /home/lgardner/git/professor/bottle.pyRN  )  s
    (   RK   RL   Rp   R�   R  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRQ  &  s   t   TwistedServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  s�   d d l  m } m } d d l m } d d l m } | �  } | j �  | j d d | j	 � | j
 | j | | | � � } | j |  j | d |  j �| j �  d  S(   Ni����(   R  R�  (   t
   ThreadPool(   t   reactort   aftert   shutdownt	   interface(   t   twisted.webR  R�  t   twisted.python.threadpoolRW  t   twisted.internetRX  R�   t   addSystemEventTriggerR.  t   Sitet   WSGIResourcet	   listenTCPR�  R�  RN  (   RI   R_  R  R�  RW  RX  t   thread_poolt   factory(    (    s&   /home/lgardner/git/professor/bottle.pyRN  5  s    	
(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRV  3  s   t   DieselServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  s3   d d l  m } | | d |  j �} | j �  d  S(   Ni����(   t   WSGIApplicationR�  (   t   diesel.protocols.wsgiRf  R�  RN  (   RI   R_  Rf  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyRN  C  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRe  A  s   t   GeventServerc           B�  s   e  Z d  Z d �  Z RS(   s�    Untested. Options:

        * `fast` (default: False) uses libevent's http server, but has some
          issues: No streaming, no pipelining, no SSL.
        * See gevent.wsgi.WSGIServer() documentation for more options.
    c         �  s�   d d l  m } m } m } t t j �  | j � sI d } t | � � n  |  j j d d  � sg | } n  |  j
 rv d  n d |  j d <|  j |  j f } | j | | |  j � �  d t j k r� d d  l } | j | j �  f d �  � n  �  j �  d  S(	   Ni����(   R�  t   pywsgiR5  s9   Bottle requires gevent.monkey.patch_all() (before import)t   fastR
   t   logR<  c         �  s
   �  j  �  S(   N(   R.  (   R0   R�   (   R  (    s&   /home/lgardner/git/professor/bottle.pyR!   [  s    (   R   R�  Ri  R5  R>   R4  R�  RC  R�   Rg   R  R�  R�  R  R�  R�   t   signalt   SIGINTR!  (   RI   R_  R�  Ri  R5  R�   RF  Rl  (    (   R  s&   /home/lgardner/git/professor/bottle.pyRN  P  s     	(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRh  I  s   t   GeventSocketIOServerc           B�  s   e  Z d  �  Z RS(   c         C�  sB   d d l  m } |  j |  j f } | j | | |  j � j �  d  S(   Ni����(   R  (   t   socketioR  R�  R�  t   SocketIOServerRC  R!  (   RI   R_  R  RF  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  `  s    (   RK   RL   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRn  _  s   t   GunicornServerc           B�  s   e  Z d  Z d �  Z RS(   s?    Untested. See http://gunicorn.org/configure.html for options. c         �  ss   d d l  m } i d |  j t |  j � f d 6�  �  j |  j � d | f �  � f d �  �  Y} | �  j �  d  S(   Ni����(   t   Applications   %s:%dRg  t   GunicornApplicationc           �  s&   e  Z �  f d  �  Z � f d �  Z RS(   c         �  s   �  S(   N(    (   RI   t   parsert   optsR�   (   R�   (    s&   /home/lgardner/git/professor/bottle.pyt   inito  s    c         �  s   �  S(   N(    (   RI   (   R_  (    s&   /home/lgardner/git/professor/bottle.pyRW  r  s    (   RK   RL   Rv  RW  (    (   R�   R_  (    s&   /home/lgardner/git/professor/bottle.pyRs  n  s   (   t   gunicorn.app.baseRr  R�  R�   R�  RJ  RC  RN  (   RI   R_  Rr  Rs  (    (   R�   R_  s&   /home/lgardner/git/professor/bottle.pyRN  h  s
    #(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRq  f  s   t   EventletServerc           B�  s   e  Z d  Z d �  Z RS(   s
    Untested c         C�  s�   d d l  m } m } y0 | j | |  j |  j f � | d |  j �Wn3 t k
 r{ | j | |  j |  j f � | � n Xd  S(   Ni����(   R�  R:  t
   log_output(   t   eventletR�  R:  R  R�  R�  R  RJ  (   RI   R_  R�  R:  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  z  s    !(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyRx  x  s   t   RocketServerc           B�  s   e  Z d  Z d �  Z RS(   s    Untested. c         C�  sC   d d l  m } | |  j |  j f d i | d 6� } | j �  d  S(   Ni����(   t   RocketR�  R'  (   t   rocketR|  R�  R�  R�   (   RI   R_  R|  R  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    %(   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR{  �  s   t   BjoernServerc           B�  s   e  Z d  Z d �  Z RS(   s?    Fast server written in C: https://github.com/jonashaag/bjoern c         C�  s*   d d l  m } | | |  j |  j � d  S(   Ni����(   RN  (   t   bjoernRN  R�  R�  (   RI   R_  RN  (    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s    (   RK   RL   Rp   RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR~  �  s   t
   AutoServerc           B�  s,   e  Z d  Z e e e e e g Z d �  Z	 RS(   s    Untested. c         C�  sR   xK |  j  D]@ } y& | |  j |  j |  j � j | � SWq
 t k
 rI q
 Xq
 Wd  S(   N(   t   adaptersR�  R�  RC  RN  R   (   RI   R_  t   sa(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s
    &(
   RK   RL   Rp   R/  R2  RV  R$  R  R�  RN  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   R�  R  R1  R*  R6  t   fapws3RO  t   gaet   twistedt   dieselR9  t   gunicornRz  t   geventSocketIOR}  R  c         K�  s�   d |  k r |  j  d d � n	 |  d f \ } }  | t j k rL t | � n  |  s] t j | S|  j �  r} t t j | |  � S| j  d � d } t j | | | <t d | |  f | � S(   s�   Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    R�  i   RL  i    s   %s.%sN(   R@  Rg   R   RF  RQ  t   isalnumRh   t   eval(   R�   Rm  RI  t   package_name(    (    s&   /home/lgardner/git/professor/bottle.pyRW  �  s    0   c         C�  sX   t  t a } z0 t j �  } t |  � } t | � r8 | S| SWd t j | � | a Xd S(   s�    Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. N(   R�   t   NORUNt   default_appR�  RW  RI  R+  (   R�   t   nr_oldR�  R=  (    (    s&   /home/lgardner/git/professor/bottle.pyt   load_app�  s    s	   127.0.0.1i�  c	         K�  s�  t  r
 d S| r~t j j d � r~z1yd }
 t j d d d d � \ } }
 t j | � x� t j j	 |
 � r=t
 j g t
 j } t j j �  } d | d <|
 | d <t j | d	 | �} x3 | j �  d k r� t j |
 d � t j | � q� W| j �  d
 k r] t j j	 |
 � r$t j |
 � n  t
 j | j �  � q] q] WWn t k
 rRn XWd t j j	 |
 � ryt j |
 � n  Xd Sy�| d k	 r�t | � n  |  p�t �  }  t |  t � r�t |  � }  n  t |  � s�t d |  � � n  x! | p�g  D] } |  j | � q�W| t k r(t j | � } n  t | t � rFt  | � } n  t | t! � rp| d | d | |	 � } n  t | t" � s�t d | � � n  | j# p�| | _# | j# s�t$ d t% t& | � f � t$ d | j' | j( f � t$ d � n  | rQt j j d � }
 t) |
 | � } | � | j* |  � Wd QX| j+ d k r^t
 j d
 � q^n | j* |  � Wnr t k
 rrnb t, t- f k
 r��  nI | s��  n  t. | d | � s�t/ �  n  t j | � t
 j d
 � n Xd S(   s�   Start a server instance. This method blocks until the server terminates.

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
     NR<  R�   s   bottle.t   suffixs   .lockt   truet   BOTTLE_LOCKFILER�  i   s   Application is not callable: %rR�  R�  s!   Unknown or unsupported server: %rs,   Bottle v%s server starting up (using %s)...
s   Listening on http://%s:%d/
s   Hit Ctrl-C to quit.

t   reloadR  (0   R�  R�  R�   R�   Rg   t   tempfilet   mkstempRJ   R�   R�  R   t
   executablet   argvR�  t
   subprocesst   Popent   pollt   utimeR+  t   sleept   unlinkt   exitRj  t   _debugR�  R>   R?  R�  RI  R�   R   t   server_namesRW  R�   R  R  RA  t   __version__R�   R�  R�  t   FileCheckerThreadRN  R1  Rk  Rl  Rh   R   (   R�   R  R�  R�  t   intervalt   reloaderR  R�   R�  RS  t   lockfilet   fdR�   R�   R�   R  t   bgcheck(    (    s&   /home/lgardner/git/professor/bottle.pyRN  �  s�      

  	 
R�  c           B�  s2   e  Z d  Z d �  Z d �  Z d �  Z d �  Z RS(   sw    Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. c         C�  s0   t  j j |  � | | |  _ |  _ d  |  _ d  S(   N(   R4  t   ThreadRc   R�  R�  Rg   R1  (   RI   R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRc   ?  s    c         C�  s[  t  j j } d �  } t �  } xq t t j j �  � D]Z } t | d d � } | d d k ri | d  } n  | r4 | | � r4 | | � | | <q4 q4 Wx� |  j	 sV| |  j
 � s� | |  j
 � t j �  |  j d k  r� d	 |  _	 t j �  n  xV t | j �  � D]B \ } } | | � s(| | � | k r� d
 |  _	 t j �  Pq� q� Wt j |  j � q� Wd  S(   Nc         S�  s   t  j |  � j S(   N(   R�  R�  R�  (   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR!   G  s    RA  R�   i����s   .pyos   .pyci����i   R�   R�  (   s   .pyos   .pyc(   R�  R�   R�  R]   R[   R   RF  R�  Rh   R1  R�  R+  R�  t   threadt   interrupt_mainR	  R�  (   RI   R�  t   mtimeR�  RI  R�   t   lmtime(    (    s&   /home/lgardner/git/professor/bottle.pyRN  E  s(    		  &		
c         C�  s   |  j  �  d  S(   N(   R�   (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt	   __enter__[  s    c         C�  s8   |  j  s d |  _  n  |  j �  | d  k	 o7 t | t � S(   NR�  (   R1  R�   Rg   R  Rj  (   RI   t   exc_typet   exc_valt   exc_tb(    (    s&   /home/lgardner/git/professor/bottle.pyt   __exit__^  s    	 
(   RK   RL   Rp   Rc   RN  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  ;  s
   			t   TemplateErrorc           B�  s   e  Z d  �  Z RS(   c         C�  s   t  j |  d | � d  S(   Ni�  (   R�   Rc   (   RI   RW   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   m  s    (   RK   RL   Rc   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  l  s   t   BaseTemplatec           B�  st   e  Z d  Z d d d d g Z i  Z i  Z d d g  d d � Z e g  d � � Z	 e d �  � Z
 d	 �  Z d
 �  Z RS(   s2    Base class and minimal API for template adapters t   tplt   htmlt   thtmlt   stplR=   c         K�  s+  | |  _  t | d � r$ | j �  n | |  _ t | d � rE | j n d |  _ g  | D] } t j j | � ^ qU |  _	 | |  _
 |  j j �  |  _ |  j j | � |  j r� |  j  r� |  j |  j  |  j	 � |  _ |  j s� t d t | � � � q� n  |  j r|  j rt d � � n  |  j |  j �  d S(   s=   Create a new template.
        If the source parameter (str or buffer) is missing, the name argument
        is used to guess a template filename. Subclasses can assume that
        self.source and/or self.filename are set. Both are strings.
        The lookup, encoding and settings parameters are stored as instance
        variables.
        The lookup parameter stores a list containing directory paths.
        The encoding parameter should be used to decode byte strings or files.
        The settings parameter contains a dict for engine-specific settings.
        Ro  R�  s   Template %s not found.s   No template specified.N(   R�   R2   Ro  Rz  R�  Rg   R�  R�   R�  R�  R(   t   settingsR�  RJ  R�  R�  R�   R�   (   RI   Rz  R�   R�  R(   R�  R    (    (    s&   /home/lgardner/git/professor/bottle.pyRc   w  s    	$!(		c         C�  s  | s t  d � d g } n  t j j | � rZ t j j | � rZ t  d � t j j | � Sx� | D]� } t j j | � t j } t j j t j j | | � � } | j | � s� qa n  t j j | � r� | Sx; |  j	 D]0 } t j j d | | f � r� d | | f Sq� Wqa Wd S(   s{    Search name in all directories specified in lookup.
        First without, then with common extensions. Return first hit. s2   The template lookup path list should not be empty.RL  s,   Absolute template path names are deprecated.s   %s.%sN(
   RY   R�  R�   t   isabsR�  R�  R�  R�   R�  t
   extensions(   Rj   R�   R�  t   spathR�  t   ext(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s     
$
!  c         G�  s;   | r, |  j  j �  |  _  | d |  j  | <n |  j  | Sd S(   sB    This reads or sets the global settings stored in class.settings. i    N(   R�  R�  (   Rj   Ra   R�   (    (    s&   /home/lgardner/git/professor/bottle.pyt   global_config�  s    c         K�  s
   t  � d S(   s�    Run preparations (parsing, caching, ...).
        It should be possible to call this again to refresh a template or to
        update settings.
        N(   t   NotImplementedError(   RI   RC  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    c         O�  s
   t  � d S(   sF   Render the template with the specified local variables and return
        a single byte or unicode string. If it is a byte string, the encoding
        must match self.encoding. This method must be thread-safe!
        Local variables may be provided in dictionaries (args)
        or directly, as keywords (kwargs).
        N(   R�  (   RI   R�   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyt   render�  s    N(   RK   RL   Rp   R�  R�  t   defaultsRg   Rc   t   classmethodR�  R�  R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  q  s   		t   MakoTemplatec           B�  s   e  Z d  �  Z d �  Z RS(   c         K�  s�   d d l  m } d d l m } | j i |  j d 6� | j d t t � � | d |  j	 | � } |  j
 r� | |  j
 d | | �|  _ n' | d |  j d	 |  j d | | � |  _ d  S(
   Ni����(   t   Template(   t   TemplateLookupR`  t   format_exceptionst   directoriesR�  t   uriR�  (   t   mako.templateR�  t   mako.lookupR�  RJ  R(   R�   R  R�   R�  Rz  R�  R�   R�  (   RI   RC  R�  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	c         O�  sJ   x | D] } | j  | � q W|  j j �  } | j  | � |  j j | �  S(   N(   RJ  R�  R�  R�  R�  (   RI   R�   R.  t   dictargt	   _defaults(    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     (   RK   RL   R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	t   CheetahTemplatec           B�  s   e  Z d  �  Z d �  Z RS(   c         K�  s~   d d l  m } t j �  |  _ i  |  j _ |  j j g | d <|  j rb | d |  j | � |  _ n | d |  j | � |  _ d  S(   Ni����(   R�  t
   searchListRz  R�  (	   t   Cheetah.TemplateR�  R4  R5  R  t   varsRz  R�  R�  (   RI   RC  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s    	c         O�  sj   x | D] } | j  | � q W|  j j j  |  j � |  j j j  | � t |  j � } |  j j j �  | S(   N(   RJ  R  R�  R�  R�   R�  R~  (   RI   R�   R.  R�  Rx  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s     (   RK   RL   R�   R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	
t   Jinja2Templatec           B�  s,   e  Z d d i  d  � Z d �  Z d �  Z RS(   c         K�  s�   d d l  m } m } d | k r1 t d � � n  | d | |  j � | � |  _ | rk |  j j j | � n  | r� |  j j j | � n  | r� |  j j	 j | � n  |  j
 r� |  j j |  j
 � |  _ n |  j j |  j � |  _ d  S(   Ni����(   t   Environmentt   FunctionLoaderR�   ss   The keyword argument `prefix` has been removed. Use the full jinja2 environment name line_statement_prefix instead.t   loader(   t   jinja2R�  R�  R�  R�  R�  R�   RJ  t   testst   globalsRz  t   from_stringR�  t   get_templateR�  (   RI   R�   R�  R�  R.  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�   �  s       	c         O�  sJ   x | D] } | j  | � q W|  j j �  } | j  | � |  j j | �  S(   N(   RJ  R�  R�  R�  R�  (   RI   R�   R.  R�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     c         C�  sQ   |  j  | |  j � } | s d  St | d � � } | j �  j |  j � SWd  QXd  S(   NR�  (   R�  R�  R�  Ro  RE   R(   (   RI   R�   R�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s
     N(   RK   RL   Rg   R�   R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s   	t   SimpleTemplatec           B�  sb   e  Z e e d d  � Z e d �  � Z e d �  � Z d d � Z	 d d � Z
 d �  Z d �  Z RS(   c         �  sh   i  |  _  |  j �  �  f d �  |  _ �  � f d �  |  _ | |  _ | rd |  j |  j |  _ |  _ n  d  S(   Nc         �  s   t  |  �  � S(   N(   R/   (   R    (   RB   (    s&   /home/lgardner/git/professor/bottle.pyR!     s    c         �  s   � t  |  �  � � S(   N(   R/   (   R    (   RB   t   escape_func(    s&   /home/lgardner/git/professor/bottle.pyR!   	  s    (   R�  R(   t   _strt   _escapet   syntax(   RI   R�  t   noescapeR�  RR   (    (   RB   R�  s&   /home/lgardner/git/professor/bottle.pyR�     s    			c         C�  s   t  |  j |  j p d d � S(   Ns   <string>R<   (   R�   R`  R�  (   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt   co  s    c         C�  s�   |  j  } | s9 t |  j d � � } | j �  } Wd  QXn  y t | � d } } Wn1 t k
 r� t d � t | d � d } } n Xt | d | d |  j �} | j	 �  } | j
 |  _
 | S(   NR�  R=   s;   Template encodings other than utf8 are no longer supported.R)   R(   R�  (   Rz  R�  R�  Ro  R/   Rf  RY   t
   StplParserR�  t	   translateR(   (   RI   Rz  R�   R(   Rt  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR`    s    	
c         K�  s0   | d  k r t d t � n  | | f | d <d  S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?t   _rebase(   Rg   RY   R�   (   RI   t   _envR�   R.  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  "  s    
c         K�  s�   | d  k r t d t � n  | j �  } | j | � | |  j k ri |  j d | d |  j � |  j | <n  |  j | j | d | � S(   NsQ   Rebase function called without arguments. You were probably looking for {{base}}?R�   R�  t   _stdout(	   Rg   RY   R�   R�  RJ  R�  R�  R�  t   execute(   RI   R�  R�   R.  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyt   _include(  s    
%c         C�  s  |  j  j �  } | j | � | j i
 | d 6| j d 6t j |  j | � d 6t j |  j | � d 6d  d 6|  j	 d 6|  j
 d 6| j d 6| j d	 6| j d
 6� t |  j | � | j d � r� | j d � \ } } d j | � | d <| 2|  j | | | � S| S(   NR�  t
   _printlistt   includet   rebaseR�  R�  R�  R�   R�   t   definedR�   R�  (   R�  R�  RJ  t   extendRM   R  R�  R�  Rg   R�  R�  R�   R�   R  R�  R�  R�   R�   (   RI   R�  R.  R�  t   subtplt   rargs(    (    s&   /home/lgardner/git/professor/bottle.pyR�  2  s    c         O�  sT   i  } g  } x | D] } | j  | � q W| j  | � |  j | | � d j | � S(   sA    Render the template using keyword arguments as local variables. R�   (   RJ  R�  R�   (   RI   R�   R.  R�  R   R�  (    (    s&   /home/lgardner/git/professor/bottle.pyR�  B  s      N(   RK   RL   R�  Rq   Rg   R�   Rr   R�  R`  R�  R�  R�  R�  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�    s   	
	t   StplSyntaxErrorc           B�  s   e  Z RS(    (   RK   RL   (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  K  s    R�  c           B�  s�   e  Z d  Z i  Z d Z e j d d � Z e d 7Z e d 7Z e d 7Z e d 7Z e d 7Z e d	 7Z e d
 7Z d Z d e Z d Z d d d � Z
 d �  Z d �  Z e e e � Z d �  Z d �  Z d �  Z d �  Z d d � Z d �  Z RS(   s    Parser for stpl templates. s�   ((?m)[urbURB]?(?:''(?!')|""(?!")|'{6}|"{6}|'(?:[^\\']|\\.)+?'|"(?:[^\\"]|\\.)+?"|'{3}(?:[^\\]|\\.|\n)+?'{3}|"{3}(?:[^\\]|\\.|\n)+?"{3}))s   |\nR�   s   |(#.*)s   |([\[\{\(])s   |([\]\}\)])sW   |^([ \t]*(?:if|for|while|with|try|def|class)\b)|^([ \t]*(?:elif|else|except|finally)\b)s?   |((?:^|;)[ \t]*end[ \t]*(?=(?:%(block_close)s[ \t]*)?\r?$|;|#))s   |(%(block_close)s[ \t]*(?=$))s   |(\r?\n)s8   (?m)^[ 	]*(\\?)((%(line_start)s)|(%(block_start)s))(%%?)s2   %%(inline_start)s((?:%s|[^'"
]*?)+)%%(inline_end)ss   <% %> % {{ }}R=   c         C�  sv   t  | | � | |  _ |  _ |  j | p. |  j � g  g  |  _ |  _ d \ |  _ |  _ d \ |  _	 |  _
 d |  _ d  S(   Ni   i    (   i   i    (   i    i    (   R/   Rz  R(   t
   set_syntaxt   default_syntaxt   code_buffert   text_buffert   linenoR�   t   indentt
   indent_modt   paren_depth(   RI   Rz  R�  R(   (    (    s&   /home/lgardner/git/professor/bottle.pyRc   n  s    c         C�  s   |  j  S(   s=    Tokens as a space separated string (default: <% %> % {{ }}) (   t   _syntax(   RI   (    (    s&   /home/lgardner/git/professor/bottle.pyt
   get_syntaxv  s    c         C�  s�   | |  _  | j �  |  _ | |  j k r� d } t t j |  j � } t t | j �  | � � } |  j	 |  j
 |  j f } g  | D] } t j | | � ^ q| } | |  j | <n  |  j | \ |  _ |  _ |  _ d  S(   Ns:   block_start block_close line_start inline_start inline_end(   R�  R@  t   _tokenst	   _re_cachet   mapR�   R�   R]   R�  t	   _re_splitt   _re_tokt   _re_inlR�   t   re_splitt   re_tokt   re_inl(   RI   R�  Rd  t   etokenst   pattern_varst   patternsR�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  z  s    	&c         C�  s�  |  j  r t d � � n  x�t r�|  j j |  j |  j  � } | r�|  j |  j  |  j  | j �  !} |  j j | � |  j  | j	 �  7_  | j
 d � r
|  j |  j  j d � \ } } } |  j j | j
 d � | j
 d � | | � |  j  t | | � d 7_  q n | j
 d � r�t d � |  j |  j  j d � \ } } } |  j j | j
 d � | | � |  j  t | | � d 7_  q n  |  j �  |  j d t | j
 d � � � q Pq W|  j j |  j |  j  � |  j �  d	 j |  j � S(
   Ns   Parser is a one time instance.i   s   
i   i   s#   Escape code lines with a backslash.t	   multilinei   R�   (   R�   R�  R�   R�  R�  Rz  R�   R�  R   R�   R~   R�  R}   RY   t
   flush_textt	   read_codeR  R�   R�  (   RI   R   R�  t   lineR�  R�   (    (    s&   /home/lgardner/git/professor/bottle.pyR�  �  s2    	 	 ".
"!
"
c      	   C�  su  d \ } } xbt  rp|  j j |  j |  j � } | sw | |  j |  j 7} t |  j � |  _ |  j | j �  | � d  S| |  j |  j |  j | j �  !7} |  j | j	 �  7_ | j
 �  \	 } } } } }	 }
 } } } | s� |  j d k r|	 s� |
 r| |	 p|
 7} q n  | r!| | 7} q | r[| } | rm| j �  j |  j d � rmt } qmq | r}|  j d 7_ | | 7} q | r�|  j d k r�|  j d 8_ n  | | 7} q |	 r�|	 d } |  _ |  j d 7_ q |
 r�|
 d } |  _ q | r
|  j d 8_ q | r,| rt } qm| | 7} q |  j | j �  | � |  j d 7_ d \ } } |  _ | s Pq q Wd  S(   NR�   i    i   i����(   R�   R�   (   R�   R�   i    (   R�   R   R�  Rz  R�   R}   t
   write_codeRP  R�   R�   R�   R�  RB  R�  Rq   R�  R�  R�  (   RI   R  t	   code_linet   commentR   R�  t   _comt   _pot   _pct   _blk1t   _blk2t   _endt   _cendt   _nl(    (    s&   /home/lgardner/git/professor/bottle.pyR  �  sV    	$'!" 	c   	      C�  s�  d j  |  j � } |  j 2| s# d  Sg  d d d |  j } } } x� |  j j | � D]� } | | | j �  !| j �  } } | r� | j | j  t t	 | j
 t � � � � n  | j d � r� | d c | 7<n  | j |  j | j d � j �  � � qU W| t | � k  r�| | } | j
 t � } | d j d � rJ| d d	  | d <n( | d j d
 � rr| d d  | d <n  | j | j  t t	 | � � � n  d d j  | � } |  j | j d � d 7_ |  j | � d  S(   NR�   i    s   \
s     s   
i����i   s   \\
i����s   \\
i����s   _printlist((%s,))s   , (   R�   R�  R�  R  R�   R�   R�   R   R�  R�   t
   splitlinesR�   RB  t   process_inlineR~   RP  R}   R�  t   countR	  (	   RI   R�  t   partst   post   nlR   R�   t   linesR`  (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s.      + )
  "c         C�  s$   | d d k r d | d Sd | S(   Ni    R�  s   _str(%s)i   s   _escape(%s)(    (   RI   t   chunk(    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s     c         C�  sX   |  j  | | � \ } } d |  j |  j } | | j �  | d 7} |  j j | � d  S(   Ns     s   
(   t   fix_backward_compatibilityR�  R�  RQ  R�  R   (   RI   R  R  R`  (    (    s&   /home/lgardner/git/professor/bottle.pyR	  �  s    c         C�  s7  | j  �  j d  d � } | r� | d d k r� t d � t | � d k rT d | f St | � d k rz d t | � | f Sd	 t | � | f Sn  |  j d k r-| j  �  r-d
 | k r-t j d | � } | r-t d � | j	 d � } |  j
 j |  j � j | � |  _
 | |  _ | | j d
 d � f Sn  | | f S(   Ni   i    R�  R�  s2   The include and rebase keywords are functions now.i   s   _printlist([base])s   _=%s(%r)s   _=%s(%r, %s)t   codings   #.*coding[:=]\s*([-\w.]+)s4   PEP263 encoding strings in templates are deprecated.s   coding*(   s   includes   rebase(   RP  R@  Rg   RY   R}   RZ   R�  R�   R�   R~   Rz  R@   R(   RE   R   (   RI   R  R  R  R   RB   (    (    s&   /home/lgardner/git/professor/bottle.pyR  �  s"    
 
 (
!	N(   RK   RL   Rp   R�  R�  R   R�  R�  R�  Rg   Rc   R�  R�  R  R�  R�  R  R  R  R	  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyR�  N  s0   







				/		c          O�  se  |  r |  d n d } | j d t � } | j d t � } t | � | f } | t k s^ t r| j d i  � } t | | � r� | t | <| rt | j | �  qqd | k s� d | k s� d | k s� d | k r� | d	 | d
 | | � t | <q| d | d
 | | � t | <n  t | s2t	 d d | � n  x |  d D] } | j
 | � q=Wt | j | � S(   s�   
    Get a rendered template as a string iterator.
    You can use a name, a filename or a template string as first parameter.
    Template rendering arguments can be passed as dictionaries
    or directly (as keyword arguments).
    i    t   template_adaptert   template_lookupt   template_settingss   
t   {t   %t   $Rz  R�  R�   i�  s   Template (%s) not foundi   N(   Rg   R�   R�  t   TEMPLATE_PATHt   idt	   TEMPLATESR�   R>   R�   R�  RJ  R�  (   R�   R.  R�  t   adapterR�  t   tplidR�  R�  (    (    s&   /home/lgardner/git/professor/bottle.pyRb    s$    
 0
 R  c         �  s   �  � f d �  } | S(   s�   Decorator: renders a template for a handler.
        The handler can control its behavior like that:

          - return a dict of template vars to fill out the template
          - return something other than a dict and the view decorator will not
            process the template, but return the handler result as is.
            This includes returning a HTTPResponse(dict) to get,
            for instance, JSON with autojson or other castfilters.
    c         �  s(   t  j �  � � �  � f d �  � } | S(   Nc          �  sg   � |  | �  } t  | t t f � rJ �  j �  } | j | � t � | � S| d  k rc t � �  � S| S(   N(   R>   R]   R9   R�  RJ  Rb  Rg   (   R�   R.  t   resultt   tplvars(   R�  Rf   t   tpl_name(    s&   /home/lgardner/git/professor/bottle.pyRP   +  s    (   RM   R  (   Rf   RP   (   R�  R+  (   Rf   s&   /home/lgardner/git/professor/bottle.pyR0  *  s    $
(    (   R+  R�  R0  (    (   R�  R+  s&   /home/lgardner/git/professor/bottle.pyR?     s    
s   ./s   ./views/s	   ../views/s   I'm a teapoti�  s   Unprocessable Entityi�  s   Precondition Requiredi�  s   Too Many Requestsi�  s   Request Header Fields Too Largei�  s   Network Authentication Requiredi�  c         c�  s+   |  ]! \ } } | d  | | f f Vq d S(   s   %d %sN(    (   R�   R  R  (    (    s&   /home/lgardner/git/professor/bottle.pys	   <genexpr>S  s    s�  
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
RL  Rv  t	   localhostR�  t   ]s   []R�  R�  R  R�  R�   R�  (  Rp   t
   __future__R    t
   __author__R�  t   __license__RK   t   optparseR   t   _cmd_parsert
   add_optiont   _optt
   parse_argst   _cmd_optionst	   _cmd_argsR  R�  t   gevent.monkeyR   t   monkeyt	   patch_allR�  R�  t   email.utilsR�  RM   R�  RG  R:  R�  R�  R�   R�  R   R�  R4  R+  RT   R   R   R)  R   R   R;  R   R   t   inspectR   t   unicodedataR   t
   simplejsonR   R   R   R.   R   R�  t   django.utils.simplejsont   version_infot   pyR  t   py25R�  R   R   R   R"   R�  RA  R�  t   http.clientt   clientt   httplibt   _threadR�  t   urllib.parseR#   R$   R�  R%   R&   R�  R'   R�  R  t   http.cookiesR*   t   collectionsR+   R9   R�  t   ioR,   t   configparserR-   R�   R?  R?   R�  RI  R�  R6   R5   t   urlparset   urllibt   Cookiet   cPickleR7   R8   R�   RU   RV   t   UserDictR:   RA   R�  R�   RC   R/   R�  RG   RH   RN   Rq   RY   R^   R�  R_   Rr   Rt   Rm  Rv   Rw   Rx   Ry   Rz   R{   R�   R�   R�   R  R�  R  R  R  Rg   R6  R7  R8  R�  t   ResponseR9  R�   R<  R!  R"  R@  RU  R�  R  R�  R]   R�   R[   R�  Rt  Rw  R  R�  R�  R�  R�  R�  R�   R�  R#  R"  R�  R�  R�  R�  R&  R�  R�  R�  R�  RX  R8  R  R  RA  R�   RZ  R\  R^  R�   RE  R/  R   RL  R�   R  R  R
  R  R$  R/  R2  R8  R;  RE  RQ  RV  Re  Rh  Rn  Rq  Rx  R{  R~  R�  R�  RW  R�  R�  RN  R�  R�  R�  R�  R�  R�  R�  R�  R�  R�  Rb  t   mako_templatet   cheetah_templatet   jinja2_templateR?  t	   mako_viewt   cheetah_viewt   jinja2_viewR$  R&  R�   R�  t	   responsest
   HTTP_CODESR	  R  Rc  R7  Rh  R5  R�   R�  R�  RI  R�  t   optR�   Rt  t   versionR�  t
   print_helpR�   R)  RF  R�   Rg  R�  R�  t   rfindRM  RP  R�   R�  R  (    (    (    s&   /home/lgardner/git/professor/bottle.pyt   <module>   s  	 �   		.	"									�w� �� �	�
$I/2�VH
Q				
				
					
	


		Z1OH�			
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
