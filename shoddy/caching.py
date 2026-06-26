'''
Lightweight lazy-caching framework with explicit dependency invalidation.

A ``cached_quantity`` is a ``functools.cached_property`` that registers itself
so that ``Cached._invalidate`` can clear it (and ``_invalidate_all`` can clear
every cached quantity) without the caller tracking individual attribute names.

This replaces the hand-rolled pattern of scattered ``self.<x> = None`` resets
and manual ``_precompute`` calls: derived quantities are computed lazily on
first access and recomputed automatically after the inputs they depend on are
invalidated.
'''

import functools


class cached_quantity(functools.cached_property):
    """A ``functools.cached_property`` that participates in bulk invalidation.

    Behaves exactly like ``functools.cached_property`` (computed once on first
    access, then stored in the instance ``__dict__`` under its own name).  The
    distinct type is what lets :class:`Cached` discover every cached quantity
    on a class via the MRO so it can clear them.
    """


class Cached:
    """Mixin providing invalidation for :class:`cached_quantity` attributes."""

    @classmethod
    def _cached_names(cls):
        """All ``cached_quantity`` attribute names defined across the MRO."""
        names = set()
        for klass in cls.__mro__:
            for key, value in vars(klass).items():
                if isinstance(value, cached_quantity):
                    names.add(key)
        return names

    def _invalidate(self, *names):
        """Drop cached values for ``names`` (all cached quantities if empty)."""
        if not names:
            names = self._cached_names()
        for name in names:
            self.__dict__.pop(name, None)
