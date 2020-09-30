from collections import defaultdict
from enum import Enum
import functools
from itertools import product
from lark import (
    Lark,
    Transformer,
    v_args,
)
from typing import (
    Optional,
    Sequence,
    Union,
)


class WhereOp(Enum):
    AND = 0
    OR = 1


@functools.total_ordering
class WhereLiteral:
    """
    Represents a literal in query selection.

    A literal is of form `attribute==value` or `attribute<>value`.
    """
    def __init__(self, attribute: str, eq_value: str, negated: bool = False):
        """
        :param attribute: attribute to be filtered by.
        :param eq_value: the value to compare attribute to.
        :param negated: If `True`, check for inequality else equality.
        """
        self.attr = attribute
        self.val = eq_value
        self.neg = negated

    def __repr__(self):
        cmp = " != " if self.neg else " = "
        return "Literal(" + self.attr + cmp + '"' + self.val + '")'

    def __str__(self):
        cmp = "!=" if self.neg else "="
        return self.attr + cmp + '"' + self.val + '"'

    def __eq__(self, other):
        return (
            self.attr == other.attr
            and self.val == other.val
            and self.neg == other.neg
        )

    def __lt__(self, other):
        return (
            (self.attr, self.val, self.neg) <
            (other.attr, other.val, other.neg)
        )

    def __hash__(self):
        return hash((self.attr, self.val, self.neg))

    def eval(self, tuple_: dict) -> bool:
        """Evaluates if the tuples fulfils the literal."""
        if self.neg:
            return tuple_[self.attr] != self.val
        else:
            pass
        return tuple_[self.attr] == self.val


@functools.total_ordering
class Where:
    """Represents the filter (WHERE clause) of a query."""
    def __init__(self, parts: Sequence[Union['Where', WhereLiteral]],
                 op: WhereOp = WhereOp.AND, is_canonical: bool = False,
                 or_exclusive: Optional[bool] = None):
        """
        :param parts: A sequence of `Where`s and literals.
        :param op: Choose disjunction (or) or conjunction (and).
        :param is_canonical: Mainly for internal use: is the filter in DNF?
        :param or_exclusive: Mainly for internal use: if all subpopulations
            selected by the filter are pairwise disjoint.
        """
        self._parts = parts
        self._op = op
        self._or_exclusive = or_exclusive
        self.is_canonical = is_canonical

    @property
    def parts(self):
        return self._parts

    @property
    def op(self):
        return self._op

    def __eq__(self, other):
        return set(self.parts) == set(other.parts) and self.op == other.op

    def __lt__(self, other):
        return (
            (self.op, sorted(self.parts)) < (other.op, sorted(other.parts))
        )

    def __hash__(self):
        return hash((*self.parts, self.op))

    def __repr__(self):
        return ("WHERE_" + self.op.name + "(" +
                ", ".join(map(lambda p: p.__repr__(), self.parts)) +
                ")")

    def __str__(self):
        return '(' + (' ' + self.op.name + ' ').join(map(str, self.parts)) + ')'

    def is_and_clause(self) -> bool:
        """Checks if the filter is an and clause (conjunction of literals)."""
        return (self.op == WhereOp.AND
                and all(map(lambda p: isinstance(p, WhereLiteral),
                            self.parts))
                and all(self.parts[i] < self.parts[i + 1]
                        for i in range(len(self.parts) - 1)))

    def is_wrapped_and_clause(self) -> bool:
        """
        Checks if the filter is a wrape and clause.

        I.e. a conjunction of literals wrapped in a else empty or `Where`.
        """
        return (self.op == WhereOp.OR
                and len(self.parts) == 1
                and isinstance(self.parts[0], Where)
                and self.parts[0].is_and_clause())

    def _simplify_and_clause(self, clause: "Where") -> Union["Where", bool]:
        """
        Returns a simplified, equivalent and clause.

        Simplifications include:
            - remove duplicate literals
            - remove negated literals if un-negated exist for the same attribute
            - check for some unsatisfiabilities and tautologies.

        :param clause: the to be simplified clause.

        :returns: the simplified clause (not necessarily a new `Where`) or
            `False` if it's obviously unsatisfiable
            `True` if it's an obvious tautology.
        """
        assert clause.is_and_clause()
        c_parts = set(clause.parts)
        assert len(c_parts) == len(clause.parts)
        if len(c_parts) == len({p.attr for p in c_parts}):
            # all attributes are unique
            return clause
        new_parts = set()
        pos_values = {}
        neg_values = defaultdict(set)
        for p in (p for p in c_parts if not p.neg):
            # process positive matches first as they are more restrictive
            if p.attr in pos_values:
                val = pos_values[p.attr]
                if p.val == val:
                    continue  # redundant literal
                else:
                    return False  # unsatisfiable: attr=val1 AND attr=val2
            pos_values[p.attr] = p.val
            new_parts.add(p)
        for p in (p for p in c_parts if p.neg):
            if p.attr in pos_values:
                val = pos_values[p.attr]
                if p.val == val:
                    return False  # unsatisfiable: attr=val1 AND attr!=val1
                else:
                    continue  # over-specified: attr=val1 AND attr!=val2
            if neg_values[p.attr]:
                values = neg_values[p.attr]
                if p.val in values:
                    continue  # redundant literal
                else:
                    pass  # fine: attr!=val1 AND attr!=val2
                    # raise NotImplemented(
                    #     "cannot handle multiple negated literals for the "
                    #     "same attribute e.g. attr != val1 AND attr != val2"
                    # )
            neg_values[p.attr].add(p.val)
            new_parts.add(p)
        assert len(set(new_parts)) == len(new_parts)
        return Where(parts=tuple(sorted(new_parts)), op=WhereOp.AND)

    def canonical(self) -> Union["Where", bool]:
        """
        Turns the filter into an equivalent filter in DNF.

        *This function has side-effects!* It can alter this instance.

        :returns: an equivalent filter in disjunctive normal form (DNF),
             `False` if it's obviously unsatisfiable, or
             `True` if it's obviously a tautology.
        """
        if self.is_canonical:
            return self
        literals = [p for p in self.parts if isinstance(p, WhereLiteral)]
        clauses = [p.canonical() for p in self.parts if isinstance(p, Where)]
        if self.op == WhereOp.AND:
            if any(c is False for c in clauses):
                # whole formula is unsatisfiable
                return False
        else:
            if any(c is True for c in clauses):
                # whole formula is tautological
                return True
        clauses = [c for c in clauses if not isinstance(c, bool)]
        clauses = list(set(c for c in clauses
                           if isinstance(c, bool) or c.parts))
        other_op_clauses = []
        for clause in clauses:
            if clause.op != self.op:
                other_op_clauses.append(clause)
            else:
                for p in clause.parts:
                    if isinstance(p, WhereLiteral):
                        literals.append(p)
                    else:
                        assert isinstance(p, Where)
                        assert p.op != self.op
                        other_op_clauses.append(p)
        if not other_op_clauses:
            if self.op == WhereOp.OR:
                ands = map(lambda l: Where(parts=(l,), op=WhereOp.AND),
                           sorted(set(literals)))
                return Where(parts=tuple(ands), op=self.op, is_canonical=True,
                             or_exclusive=self._or_exclusive)
            else:
                and_ = Where(parts=tuple(sorted(set(literals))), op=self.op)
                and_s = self._simplify_and_clause(and_)
                if and_s is False:
                    # print("Dropping unsatisfiable clause: %s" % repr(and_))
                    return False
                return Where(parts=(and_s,), op=WhereOp.OR, is_canonical=True,
                             or_exclusive=self._or_exclusive)
        if self.op == WhereOp.OR:
            assert all(
                all(
                    isinstance(p, WhereLiteral) for p in c.parts
                ) for c in other_op_clauses
            )
            literal_clauses = map(lambda l: Where(parts=(l,), op=WhereOp.AND),
                                  set(literals))
            parts = sorted({*literal_clauses, *other_op_clauses})
            return Where(parts=tuple(parts), op=WhereOp.OR, is_canonical=True,
                         or_exclusive=self._or_exclusive)
        else:
            new_and_clauses = []
            seen_combinations = set((l,) for l in literals)

            if len(other_op_clauses) <= 1:
                part_combinations = [(p,) for c in other_op_clauses
                                     for p in c.parts]
            else:
                part_combinations = product(*map(lambda c: c.parts,
                                                 other_op_clauses))
            for com in part_combinations:
                inner_literals = [
                    l for p in
                    (list(c.parts) if isinstance(c, Where) else [c]
                     for c in com)
                    for l in p

                ]
                parts = tuple(sorted(set(inner_literals + literals)))
                if parts not in seen_combinations:
                    seen_combinations.add(parts)
                    new_and_clauses.append(Where(parts=parts, op=WhereOp.AND))
            new_and_clauses_filtered = set()
            for c in new_and_clauses:
                c = self._simplify_and_clause(c)
                if c:
                    new_and_clauses_filtered.add(c)
                # else:
                    # print("Dropping unsatisfiable clause: %s" % repr(c))
                new_and_clauses_filtered.add(c)
            if not new_and_clauses_filtered:
                return False
            new_parts = tuple(sorted(new_and_clauses_filtered))
            return Where(parts=new_parts, op=WhereOp.OR, is_canonical=True,
                         or_exclusive=self._or_exclusive)

    def or_exclusive(self) -> bool:
        """
        Checks if all selected subpopulations are pairwise disjoint.

        :raises ValueError: if this filter ins not canonical.
        """
        if self._or_exclusive is not None:
            return self._or_exclusive
        if not self.is_canonical:
            raise ValueError("Can only check or_exclusive() for canonical "
                             "queries.")
        for i in range(len(self.parts)):
            i_j_disjoint = True  # in case i == len(self.parts) + 1
            for j in range(i + 1, len(self.parts)):
                i_j_disjoint = False
                for lit_i in self.parts[i].parts:
                    for lit_j in self.parts[j].parts:
                        if (lit_i.attr == lit_j.attr
                                and ((lit_i.val != lit_j.val)
                                     ^ (lit_i.neg != lit_j.neg))):
                            i_j_disjoint = True
                            break
                    if i_j_disjoint:
                        break
                if i_j_disjoint:
                    break
            if not i_j_disjoint:
                self._or_exclusive = False
                return False
        self._or_exclusive = True
        return True

    def eval(self, tuple_: dict) -> bool:
        """Checks if the given tuple satisfies the selection."""
        reduce = all if self._op == WhereOp.AND else any
        return reduce(map(lambda x: x.eval(tuple_), self._parts))


WHERE_GRAMMAR = r"""
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS

    start: or_clause                         -> start

    or_clause: and_clause                    -> id
             | and_clause (_OR and_clause)+  -> or_

    and_clause: atom                         -> id
              | atom (_AND atom)+            -> and_

    atom: NAME _EQUAL value                  -> eq
        | NAME _NOT_EQUAL value              -> neq
        | "(" or_clause ")"                  -> id

    value: QUOTED_STR                        -> quoted_val
         | NAME                              -> id
         | NUMBER                            -> id

    QUOTED_STR: /\".*\"(?<!\\\")|\'.*\'(?<!\\\')/

    _OR: /(?<=\s)OR(?=\s)/
    _AND: /(?<=\s)AND(?=\s)/

    _EQUAL: "=="
          | "="

    _NOT_EQUAL: "!="
              | "<>"

    %ignore WS
"""


@v_args(inline=True)
class _T(Transformer):
    def start(self, clause):
        if not isinstance(clause, Where):
            return Where(parts=(clause,))
        return clause

    def or_(self, *clauses):
        # print('or', clauses)
        return Where(parts=clauses, op=WhereOp.OR)
        # raise NotImplementedError('WHERE currently only supports AND')

    def and_(self, *clauses):
        # print('and', clauses)
        return Where(parts=clauses, op=WhereOp.AND)

    def eq(self, name, value):
        # print('eq', name, value)
        return WhereLiteral(attribute=str(name), eq_value=str(value))

    def neq(self, name, value):
        # print('neq', name, value)
        return WhereLiteral(attribute=str(name), eq_value=str(value),
                            negated=True)

    def quoted_val(self, val):
        return str(val)[1:-1]

    def id(self, val):
        return val


parse_where = Lark(WHERE_GRAMMAR, parser='lalr', transformer=_T()).parse


class Query:
    """Represents a whole query (selection and grouped attributes)."""
    def __init__(self, groups: Sequence[str], where: Union[bool, Where] = True):
        """
        :param groups: the attributes to group by.
        :param where: the selection. Use `True` to indicate no selection.
        """
        assert groups
        self.groups = groups
        self.where = where
        self.is_canonical = False

    def __hash__(self):
        return hash((*self.groups, self.where))

    def __eq__(self, other: "Query") -> bool:
        return (set(self.groups) == set(other.groups)
                and self.where == other.where)

    def canonical(self) -> "Query":
        """Turn the query into canonical form: sorted groups & where in DNF."""
        if self.is_canonical:
            return self
        if isinstance(self.where, bool):
            q = Query(sorted(self.groups), self.where)
        else:
            q = Query(sorted(self.groups), self.where.canonical())
        q.is_canonical = True
        return q

    def __str__(self):
        return "GROUP BY {} WHERE {}".format(
            ", ".join(self.groups), str(self.where).lower()
        )


class QueryRewriteEngine:
    # TODO: out of scope
    @staticmethod
    def optimize(q: Query, result_cache: 'ResultCache') -> Query:
        pass


if __name__ == '__main__':
    parser = Lark(WHERE_GRAMMAR, parser='lalr', transformer=_T())
    res = parser.parse("a=a AND b=b AND c=c OR d=d OR e=e")
    res2 = parser.parse("name=\"test est\" AND (a=b OR c==d)")
    res3 = Where(parts=(res, res2))
    print(res3)
    print(res3.canonical())

    print(parser.parse("(a=b OR c=a) AND d=d"))

    res4 = parser.parse("a=a")
    print(res4)
    print(res4.canonical().__repr__(), '<-------------')
    print(parser.parse("a=a AND b=b OR c=c").canonical().__repr__())
    print(parser.parse("a=a OR b=b AND c=c"))
    print(Lark(WHERE_GRAMMAR, parser='lalr').parse("a=a AND b=b OR c=c"))
    print(Lark(WHERE_GRAMMAR, parser='lalr').parse("a=a OR b=b AND c=c"))

    w = Where(
        parts=(
            WhereLiteral('a', 'a'),
            WhereLiteral('b', 'a'),
        ),
        op=WhereOp.AND
    )
    w = Where(parts=(w,), op=WhereOp.OR)
    print(repr(w))
    print(repr(w.canonical()))
    w2 = Where(parts=(w, WhereLiteral('c', 'c')), op=WhereOp.AND)
    print(repr(w2))
    print(repr(w2.canonical()))
