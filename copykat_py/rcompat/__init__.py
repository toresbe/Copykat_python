"""Pieces of R that CopyKAT-R's results depend on, reproduced for the ``r`` profile.

``rng``: R's default random number generator (Mersenne-Twister seeded by ``set.seed``,
inversion normals, ``exp_rand`` and ``rgamma``), so step 5 draws R's own numbers.
``rstats``: R's long-double ``sum``/``mean`` and ``ks.test``'s statistic.
"""
