import random
from basictypes import Atom, Variable, Constant, Predicate
from copy import deepcopy
from copy import copy
import numpy as np


# Creates copy of our dictionary class defined in mapping.py to
# cirumvent circular import statements, as well as to define
# dictionary of term/predicate mappings for purposes of
# individual rule translations vie knowledgebase.transRule() method.
class dictionary(dict):
    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


def getKeys(dictName):
    keysList = list(dictName.keys())
    return keysList


def getValues(dictName):
    valList = list(dictName.values())
    return valList


# defines a rule with a head and a list of atoms in the body
class Rule:
    def __init__(self, head: Atom, body: list[Atom]) -> None:
        self.head = head
        self.body = body
        self.length = len(self.body)
        self.update_vars()

    def __len__(self):
        return self.length

    def __str__(self) -> str:
        bodystr = ''
        for i in range(len(self.body)):
            bodystr += str(self.body[i])
            if i < len(self.body)-1:
                bodystr += ", "
        if len(self.body) != 0:
            return f"{self.head} :- " + bodystr
        else:
            return f"{self.head}"

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

    def __hash__(self) -> int:
        return hash(str(self))

    def __deepcopy__(self, memodict={}):
        return Rule(deepcopy(self.head), [deepcopy(x) for x in self.body])

    def __copy__(self):
        return Rule(copy(self.head), [copy(x) for x in self.body])

    def setHead(self, newHead: Atom):
        self.head = newHead

    def setBody(self, newBody: list[Atom]):
        self.body = newBody
        self.update_vars()

    def update_vars(self):
        body_args = set()
        for atom in self.body:
            for arg in atom.arguments:
                if isinstance(arg, Variable):
                    body_args.add(arg.name)
        # JDH 4/28/24: Changed this to record the actual variables instead of only their names
        self.vars = {var for var in self.head.arguments if isinstance(
            var, Variable)}.union(body_args)

    def takeMaxArity(self):
        maxArity = 0
        headAtom = self.head
        if headAtom.arity > maxArity:
            maxArity = headAtom.arity
        if self.length > 0:
            for i in range(self.length):
                if self.body[i].arity > maxArity:
                    maxArity = self.body[i].arity
        return maxArity

    # Not using this functionality
    # def takeRuleMapping(self) -> dict:
    #     headAtom = self.head
    #     maxArity = self.takeMaxArity()+1
    #     predVal = 0
    #     constVal = 0
    #     varVal = 0
    #     predDetected = []
    #     constDetected = []
    #     varDetected = []
    #     predDict = dictionary()
    #     constDict = dictionary()
    #     varDict = dictionary()
    #     for i in range(maxArity):
    #         seqLevel = 0
    #         headAtom = self.head
    #         if headAtom.arity == i:
    #             if headAtom.predicate.name not in predDetected:
    #                 predDetected.append(headAtom.predicate.name)
    #                 predDict.key = headAtom.predicate.name
    #                 predDict.value = 'p' + str(i) + '_' + str(seqLevel)
    #                 predDict.add(predDict.key, predDict.value)
    #                 seqLevel += 1
    #             for j in range(len(headAtom.arguments)):
    #                 if type(headAtom.arguments[j]) == Constant:
    #                     if headAtom.arguments[j] not in constDetected:
    #                         constDetected.append(headAtom.arguments[j])
    #                         constDict.key = str(headAtom.arguments[j])
    #                         constString = constDict.key
    #                         if constString[-1] == ')':
    #                             constDict.key = constString[0:len(
    #                                 constString)-1]
    #                         constDict.value = 'c' + str(constVal)
    #                         constDict.add(constDict.key, constDict.value)
    #                         constVal += 1
    #                 elif type(headAtom.arguments[j]) == Variable:
    #                     if headAtom.arguments[j] not in varDetected:
    #                         varDetected.append(headAtom.arguments[j])
    #                         varDict.key = str(headAtom.arguments[j])
    #                         varString = varDict.key
    #                         if varString[-1] == ')':
    #                             varDict.key = varString[0:len(varString)-1]
    #                         varDict.value = 'X' + str(varVal)
    #                         varDict.add(varDict.key, varDict.value)
    #                         varVal += 1
    #         if self.length > 0:
    #             for j in range(self.length):
    #                 atom = self.body[j]
    #                 if atom.arity == i:
    #                     if atom.predicate.name not in predDetected:
    #                         predDetected.append(atom.predicate.name)
    #                         predDict.key = atom.predicate.name
    #                         predDict.value = 'p' + str(i) + '_' + str(seqLevel)
    #                         predDict.add(predDict.key, predDict.value)
    #                         seqLevel += 1
    #                     for k in range(len(atom.arguments)):
    #                         if type(atom.arguments[k]) == Constant:
    #                             if str(atom.arguments[k]) not in constDetected:
    #                                 constDetected.append(atom.arguments[k])
    #                                 constDict.key = str(atom.arguments[k])
    #                                 constDict.value = 'c' + str(constVal)
    #                                 constDict.add(
    #                                     constDict.key, constDict.value)
    #                                 constVal += 1
    #                         elif type(atom.arguments[k]) == Variable:
    #                             if str(atom.arguments[k]) not in varDetected:
    #                                 varDetected.append(atom.arguments[k])
    #                                 varDict.key = str(atom.arguments[k])
    #                                 varString = varDict.key
    #                                 if varString[-1] == ')':
    #                                     varDict.key = varString[0:len(
    #                                         varString)-1]
    #                                 varDict.value = 'X' + str(varVal)
    #                                 varDict.add(varDict.key, varDict.value)
    #                                 varVal += 1
    #     finalDict = dictionary()
    #     finalDict.update(predDict)
    #     finalDict.update(constDict)
    #     finalDict.update(varDict)
    #     return finalDict
    #
    # def transRule(self, mappingDict: dict):
    #     headAtom = self.head
    #     headAtomArity = headAtom.arity
    #     headAtomNewPred = mappingDict.get(headAtom.predicate.name)
    #     argList = []
    #     for i in range(len(headAtom.arguments)):
    #         checkString = str(headAtom.arguments[i])
    #         if checkString[-1] == ')':
    #             checkString = checkString[0:len(checkString)-1]
    #         newArg = mappingDict.get(checkString)
    #         argList.append(newArg)
    #     newHeadPredicate = Predicate(headAtomArity, headAtomNewPred)
    #     newHeadAtom = Atom(newHeadPredicate, argList)
    #     self.head = newHeadAtom
    #
    #     if self.length > 0:
    #         mappedAtomBody = []
    #         for i in range(self.length):
    #             atom = self.body[i]
    #             curAtomArity = atom.arity
    #             newAtomPred = mappingDict.get(atom.predicate.name)
    #             newAtomArgList = []
    #             for j in range(len(atom.arguments)):
    #                 checkString = str(atom.arguments[j])
    #                 if checkString[-1] == ')':
    #                     checkString = checkString[0:len(checkString)-1]
    #                 newArg = mappingDict.get(checkString)
    #                 newAtomArgList.append(newArg)
    #             newAtomPredicate = Predicate(curAtomArity, newAtomPred)
    #             newAtom = Atom(newAtomPredicate, newAtomArgList)
    #             mappedAtomBody.append(newAtom)
    #         self.body = mappedAtomBody
    #
    # def transRuleBackToKB(self, mappingDict: dict):
    #     headAtom = self.head
    #     headAtomArity = headAtom.arity
    #     keyList = list(mappingDict.keys())
    #     valList = list(mappingDict.values())
    #     curHeadAtomPred = headAtom.predicate.name
    #     originalHeadAtomPredPos = valList.index(curHeadAtomPred)
    #     originalHeadAtomPred = keyList[originalHeadAtomPredPos]
    #     argList = []
    #     for i in range(len(headAtom.arguments)):
    #         checkString = headAtom.arguments[i]
    #         if checkString[-1] == ')':
    #             checkString = checkString[0:len(checkString)-1]
    #         origArgPos = valList.index(checkString)
    #         origArg = keyList[origArgPos]
    #         argList.append(origArg)
    #     origHeadAtomPredicate = Predicate(headAtomArity, originalHeadAtomPred)
    #     originalHeadAtom = Atom(origHeadAtomPredicate, argList)
    #     self.head = originalHeadAtom
    #
    #     if self.length > 0:
    #         origAtomBody = []
    #         for i in range(self.length):
    #             curAtom = self.body[i]
    #             curAtomArity = curAtom.arity
    #             curAtomPred = curAtom.predicate.name
    #             originalAtomPredPos = valList.index(curAtomPred)
    #             originalAtomPred = keyList[originalAtomPredPos]
    #             argList = []
    #             for j in range(len(curAtom.arguments)):
    #                 checkString = curAtom.arguments[j]
    #                 if checkString[-1] == ')':
    #                     checkString = checkString[0:len(checkString)-1]
    #                 origArgPos = valList.index(checkString)
    #                 origArg = keyList[origArgPos]
    #                 argList.append(origArg)
    #             origAtomPred = Predicate(curAtomArity, originalAtomPred)
    #             originalAtom = Atom(origAtomPred, argList)
    #             origAtomBody.append(originalAtom)
    #         self.body = origAtomBody


# knowledge base; list of rules with methods to manipulate and add rules
class KnowledgeBase:
    def __init__(self, rules: list[Rule] = None) -> None:
        if rules is None:
            self.rules = []
        else:
            self.rules = rules
        self.length = len(self.rules)
        self.rule_by_pred = {}
        # self.pred_set = set()
        for rule in self.rules:
            if rule.head.predicate in self.rule_by_pred:
                self.rule_by_pred[rule.head.predicate].append(rule)
            else:
                self.rule_by_pred[rule.head.predicate] = [rule]

    def __len__(self):
        return self.length

    def try_index_pred(self, index_rule):
        if index_rule in self.rule_by_pred:
            return self.rule_by_pred[index_rule]
        else:
            return self.rules

    def addrule(self, rule):
        self.rules.append(rule)
        self.length += 1
        # if KnowledgeBase.is_fact(rule):
        #     self.facts.append(rule)
        if rule.head.predicate in self.rule_by_pred:
            self.rule_by_pred[rule.head.predicate].append(rule)
        else:
            self.rule_by_pred[rule.head.predicate] = [rule]

    # def get_facts(self):
    #     return self.facts

    def print(self):
        for rule in self.rules:
            print(str(rule))

    def get_pred_list(self):
        return list(self.rule_by_pred.keys())

    def rename(self):
        predicates = {"p" + str(i) for i in range(10)}
        variables = {"X" + str(i) for i in range(10)}
        constants = {"a" + str(i) for i in range(100)}
        names = {}
        try:
            for i in range(len(self.rules)):
                if self.rules[i].head.predicate.name not in names:
                    p = predicates.pop()
                    names[self.rules[i].head.predicate.name] = p
                    self.rules[i].head.predicate.name = p
                else:
                    self.rules[i].head.predicate.name = names[self.rules[i].head.predicate.name]
                for j in range(len(self.rules[i].head.arguments)):
                    if self.rules[i].head.arguments[j].name not in names:
                        if isinstance(self.rules[i].head.arguments[j], Constant):
                            c = constants.pop()
                            names[self.rules[i].head.arguments[j].name] = c
                            self.rules[i].head.arguments[j].name = c
                        else:
                            v = variables.pop()
                            names[self.rules[i].head.arguments[j].name] = v
                            self.rules[i].head.arguments[j].name = v
                    else:
                        self.rules[i].head.arguments[j].name = names[self.rules[i].head.arguments[j].name]

                for arg_num in range(len(self.rules[i].body)):
                    if self.rules[i].body[arg_num].predicate.name not in names:
                        p = predicates.pop()
                        names[self.rules[i].body[arg_num].predicate.name] = p
                        self.rules[i].body[arg_num].predicate.name = p
                    else:
                        self.rules[i].body[arg_num].predicate.name = names[self.rules[i].body[arg_num].predicate.name]
                    for j in range(len(self.rules[i].body[arg_num].arguments)):
                        if self.rules[i].body[arg_num].arguments[j].name not in names:
                            if isinstance(self.rules[i].body[arg_num].arguments[j], Constant):
                                c = constants.pop()
                                names[self.rules[i].body[arg_num].arguments[j].name] = c
                                self.rules[i].body[arg_num].arguments[j].name = c
                            else:
                                v = variables.pop()
                                names[self.rules[i].body[arg_num].arguments[j].name] = v
                                self.rules[i].body[arg_num].arguments[j].name = v
                        else:
                            self.rules[i].body[arg_num].arguments[j].name = names[self.rules[i].body[arg_num].arguments[j].name]
            print("success")
        except KeyError:
            print("Translate failed")
            print("Too many different c/p/v")


# defines path in with a node, root, rule, and depth
class Path:
    def __init__(self, node: Atom, parent, rule: Rule, depth: int, all_goals=None) -> None:
        self.node: Atom = node          # the goal to prove, maybe don't need anymore?
        self.parent = parent            # the parent path node
        # the target goal that was resolved to produce the current node
        self.target: Atom = None
        self.rule = rule                # the rule applied to produce this node
        if all_goals is None:           # all subgoals for the current node
            self.all_goals = []
        else:
            self.all_goals = all_goals
        if self.all_goals == []:             # a list of integers indicating the level at which each goal was introduced
            self.goal_starts = []
        else:
            # fill with as many 0's as there are initial goals
            self.goal_starts = [0]*len(all_goals)
        self.depth: int = depth         # the length of the path
        self.node_level: int = 0   # the level that the parent goal node was introduced on
        # self.path_success: bool = False # overall success of the path
        # the original level of the goal on which the search path failed, -1 ia no fail
        self.fail_level = -1
        self.examples = set()           # TODO: consider removing this
        self.s = 0                      # the count of successes
        self.f = 0                      # the count of failures

    def get_leaf(self, rule, node):
        """ Creates a new child path object with self as the 'patent' and depth+1
        """
        return Path(node, self, rule, self.depth+1)

    def make_child(self, target, rule, all_goals, num_new_goals):
        """ Create a new child path object. This is an improved version of get_leaf
        that stores information needed to properly score the nodes in the search """
        c = Path(None, self, rule, self.depth+1)
        c.target = target
        c.rule = rule
        c.all_goals = all_goals
        # TODO: test the following thoroughly
        # node_level is the level of the target goal
        # the target will be the first goal, thus its level is the first goal start
        c.node_level = self.goal_starts[0]

        # replace the following with the above code
        # if num_new_goals > 0:
            # this is the level of the target goal
            # c.node_level = self.goal_starts[0]
        # else:             # when we solved the previous goal, we use the level of its parent goal instead
           # c.node_level = self.node_level

        # removes first goal, adds body of rule
        c.goal_starts = [c.depth]*num_new_goals + self.goal_starts[1:]
        return c

    def get_depth(self):
        return self.depth

    def inc_fail(self):
        self.f += 1

    def inc_success(self):
        self.s += 1  # discounting based on depth?

    def set_node(self, node):
        self.node = node

    def get_sf(self):
        """ Returns the ratio of successes over all nodes used to prove the goal with the rule.
        """
        return self.s / (self.f + self.s)

    def get_score(self):
        """ Returns a score for the node. """
        if self.s > 0:
            return 1
        else:
            return 0

    def print_rule_path(self):
        """ Prints all rules along the path, from root to leaf. """
        if self.parent is not None:
            self.parent.print_rule_path()
        if self.target is not None:
            print(str(self.target) + ": " + str(self.rule))

    def push_root(self):
        # TODO: should this be recursive? If so, I have a few commented lines at end of function
        self.parent.s += self.s
        self.parent.f += self.f
        # if not (self.root is None):
        #     self.root.push_root()


def generate_rule_head(c_list, v_list, p_list, only_const):
    """
    Generates the head of a random rule, choosing from a list of predicates, variables, and constants
    If only_const is true, the head will always be ground (used to generate a fact).
    :param c_list:
    :param v_list:
    :param p_list:
    :param only_const:
    :return:
    """
    p: Predicate = random.choice(p_list)
    args = []
    var_pool = set()
    const_pool = set()
    for _ in range(p.arity):
        # TODO: 0.5 is probably too high a threshold for a constant in the head
        if random.random() >= 0.5 or only_const:
            c = random.choice(c_list)
            args.append(deepcopy(c))
            const_pool.add(c.name)
        else:
            v = deepcopy(random.choice(v_list))
            args.append(v)
            var_pool.add(v.name)
    return Atom(p, args), var_pool, const_pool


def generate_rule_body_arg(c_list, v_list, p_list, var_pool, const_pool):
    c_odds = 0.4
    p = random.choice(p_list)
    args = []
    for i in range(p.arity):
        if i == 0 and (const_pool or var_pool):
            if const_pool:
                if var_pool:
                    if random.random() >= c_odds:
                        v = random.choice(list(var_pool))
                        args.append(Variable(v))
                        var_pool.add(v)
                    else:
                        c = random.choice(list(const_pool))
                        args.append(Constant(c))
                else:
                    c = random.choice(list(const_pool))
                    args.append(c)
                    const_pool.add(c)
            elif var_pool:
                v = random.choice(list(var_pool))
                args.append(Variable(v))
                var_pool.add(v)

        else:
            if random.random() >= c_odds:
                v = random.choice(v_list)
                args.append(v)
                var_pool.add(v.name)
            else:
                c = random.choice(c_list)
                args.append(c)
                var_pool.add(c.name)
    random.shuffle(args)
    a = Atom(p, args)
    return a


def generate_random_KB(vocab, min_rules):  # max rule length = 4
    num_rules = 0
    KB = KnowledgeBase([])
    while num_rules < min_rules:
        rule_len = np.random.choice(
            [0, 1, 2, 3, 4], p=[0.8, 0.08, 0.07, 0.03, 0.02])
        only_const = False
        if rule_len == 0:
            only_const = True
        # TODO: The variables of the body should be used as the var_pool for the head...
        head, var_pool, const_pool = generate_rule_head(vocab.constants, vocab.variables,
                                                        vocab.predicates, only_const)

        body = []
        for i in range(rule_len):
            arg = generate_rule_body_arg(vocab.constants, vocab.variables, vocab.predicates,
                                         var_pool, const_pool)
            body.append(arg)
            random.shuffle(body)
        rule = Rule(head, body)
        if rule not in KB.rules:  # Could make it a set for performance boost
            KB.addrule(rule)
            num_rules += 1
    return KB


if __name__ == "__main__":
    pass
