sf.decompositions
=================

.. currentmodule:: strawberryfields.decompositions

.. warning::

    Unless you are a Strawberry Fields developer, you likely do not need
    to use these functions directly.

    See the :ref:`decomposition <decompositions>` operations for
    details on embedding states, graphs, and unitaries within a photonic
    quantum circuit.

.. automodapi:: strawberryfields.decompositions
    :no-heading:
    :include-all-objects:
    :skip: block_diag, sqrtm, polar, schur, sympmat, xpxp_to_xxpp, adj_scaling

.. _rectangular:

Rectangular interferometer decomposition
------------------------------------------

:func:`strawberryfields.decompositions.rectangular` describes the rectangular
interferometer decomposition, its phase convention and returned components.

.. _williamson:

Williamson decomposition
--------------------------

See :func:`strawberryfields.decompositions.williamson` for the input matrix
requirements and the diagonal and symplectic matrices it returns.

.. _bloch_messiah:

Bloch-Messiah decomposition
-----------------------------

:func:`strawberryfields.decompositions.bloch_messiah` documents the decomposition
of a symplectic transformation into passive transformations and squeezing.
