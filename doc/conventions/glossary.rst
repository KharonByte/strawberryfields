.. _conventions:

Conventions and glossary
==========================

.. _opcon:

Quadrature conventions
----------------------

Strawberry Fields uses dimensionless quadrature coordinates with a configurable
value of :math:`\hbar`; the default is :math:`\hbar=2`. In terms of the annihilation
operator :math:`\hat a`, the quadrature operators are

.. math::

    \hat x = \sqrt{\hbar/2}(\hat a + \hat a^\dagger),
    \qquad
    \hat p = -i\sqrt{\hbar/2}(\hat a - \hat a^\dagger).

They obey :math:`[\hat x,\hat p]=i\hbar`. See :doc:`/introduction/states` and
:doc:`/introduction/ops` for state preparations and operations in these coordinates.

.. _homodyne:

Homodyne measurement
----------------------

Homodyne detection measures the quadrature
:math:`\hat x_\phi=\hat x\cos\phi+\hat p\sin\phi`.
:class:`~strawberryfields.ops.MeasureHomodyne` documents the phase parameter,
returned measurement and postselection interface.

Compiler terminology
--------------------

.. glossary::

   equivalent circuit
      A circuit expressing the same quantum operation using a different sequence
      of gates. Circuit compilation can replace an operation by a decomposition
      supported by the target backend.

   circuit class
      A compiler's specification of allowed operations and circuit structure.
      See :class:`~strawberryfields.compilers.Compiler` for how these constraints
      are represented and checked.
