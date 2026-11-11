"""Multiple-choice technical screening (Milestone 3 extension).

Why this exists, stated plainly: `interview.score_answer()` grades *how* an answer is
built — structure, concrete detail, whether the skill was addressed. It cannot tell
whether the answer is technically **correct**. A fluent, well-organised wrong answer
scores well. That limitation is written into the README, and this module is the fix for
it: an MCQ has exactly one defensible right answer, so the score is objective knowledge,
not presentation.

The two are complementary and are reported separately, never averaged into one number:

    MCQ          does this person know the material?      objective, auto-scored
    free text    can this person explain and apply it?    rubric, judgement-based

Questions are selected from the same Milestone 2 match the interview kit uses, so a
candidate is tested on the skills *this role* requires and *their resume* claims — not a
generic quiz. Difficulty follows the seniority the parser computed.

Every option set is shuffled with a seed derived from the candidate and question, so the
correct answer is not always in the same position, and the same candidate always sees the
same test (re-running a screen cannot quietly change the result).

Honest limitation: the bank is hand-written and finite (see BANK_SIZE). It covers the
skills in the four evaluation roles well and thins out beyond them. A candidate whose
skills fall outside the bank gets fewer questions, and `coverage` in the result reports
exactly that rather than padding the test with irrelevant items.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .matching import match
from .taxonomy import CATEGORIES, TAXONOMY

PASS_MARK = 60.0          # percent, per the standard screening convention
DIFFICULTY_BY_SENIORITY = {
    "Entry level": "easy", "Junior": "easy", "Mid level": "medium",
    "Senior": "hard", "Principal / Lead": "hard", "Unclassified": "medium",
}
# a test is not all one level: mostly the target tier, with a step either side
MIX_BY_TARGET = {
    "easy":   [("easy", 0.70), ("medium", 0.30), ("hard", 0.00)],
    "medium": [("easy", 0.25), ("medium", 0.55), ("hard", 0.20)],
    "hard":   [("easy", 0.10), ("medium", 0.40), ("hard", 0.50)],
}
SECONDS_PER_QUESTION = {"easy": 45, "medium": 70, "hard": 100}


# --------------------------------------------------------------------------- #
# the bank
#
# Each entry: (skill, difficulty, question, options, correct_option, explanation)
# `correct_option` is the option *text*, not an index — so reordering the options
# list while editing can never silently change which answer is right.
# --------------------------------------------------------------------------- #
Item = Tuple[str, str, str, List[str], str, str]

BANK: List[Item] = [
    # ---------------- Programming languages ----------------
    ("Python", "easy", "What does a list comprehension `[x*2 for x in nums]` return?",
     ["A new list with each element doubled",
      "The original list, modified in place",
      "A generator that must be consumed with next()",
      "A dictionary keyed by the original values"],
     "A new list with each element doubled",
     "A list comprehension always builds and returns a new list. The original is untouched. "
     "A generator would use parentheses instead of brackets."),

    ("Python", "medium", "What does Python's Global Interpreter Lock (GIL) prevent?",
     ["Two threads running Python bytecode at the same time in one process",
      "Two processes accessing the same file",
      "All race conditions in multithreaded code",
      "Memory leaks in long-running programs"],
     "Two threads running Python bytecode at the same time in one process",
     "The GIL serialises bytecode execution within a process, so CPU-bound threads do not run "
     "in parallel — multiprocessing sidesteps it. It does not prevent race conditions: an "
     "operation can still be interrupted between bytecodes."),

    ("Python", "medium", "Why is a mutable object like `[]` a dangerous default argument?",
     ["The same object is reused across every call that omits the argument",
      "Python raises a TypeError at definition time",
      "It is copied on every call, which is slow",
      "It silently becomes a tuple"],
     "The same object is reused across every call that omits the argument",
     "Default arguments are evaluated once, when the function is defined. Mutating that "
     "default leaks state between calls. The usual fix is a `None` default and building the "
     "list inside the function."),

    ("Python", "hard", "What is the practical difference between `__str__` and `__repr__`?",
     ["`__str__` targets end users; `__repr__` targets developers and should be unambiguous",
      "`__repr__` must return bytes, `__str__` returns text",
      "`__str__` is called by the interpreter, `__repr__` by print()",
      "They are aliases — defining either is enough for both"],
     "`__str__` targets end users; `__repr__` targets developers and should be unambiguous",
     "`print()` uses `__str__`; the REPL and containers use `__repr__`. Defining only "
     "`__repr__` does give you a `__str__` fallback, but not the other way round."),

    ("Java", "easy", "What does the `final` keyword do on a variable?",
     ["Prevents the variable from being reassigned after initialisation",
      "Makes the object it points to immutable",
      "Marks the variable for garbage collection",
      "Makes the variable visible to every class"],
     "Prevents the variable from being reassigned after initialisation",
     "`final` binds the reference, not the object. A `final List` cannot point elsewhere, "
     "but you can still call `add()` on it."),

    ("Java", "medium", "When would you choose an abstract class over an interface?",
     ["When you need shared state and a constructor alongside shared behaviour",
      "When several unrelated classes must implement the same methods",
      "When you want multiple inheritance of behaviour",
      "When the type will never be subclassed"],
     "When you need shared state and a constructor alongside shared behaviour",
     "An abstract class can hold fields and a constructor, so it suits an 'is-a' hierarchy "
     "with common state. Interfaces give a contract that unrelated classes can implement, and "
     "a class may implement many."),

    ("Java", "hard", "Why must you override `hashCode()` whenever you override `equals()`?",
     ["Hash-based collections will otherwise fail to find objects that are equal",
      "The compiler rejects the class without it",
      "`equals()` calls `hashCode()` internally",
      "Serialisation requires both to be defined"],
     "Hash-based collections will otherwise fail to find objects that are equal",
     "`HashMap` and `HashSet` locate a bucket by hash first, then compare with `equals()`. "
     "Two equal objects with different hashes land in different buckets and never match."),

    ("JavaScript", "easy", "What does `===` do that `==` does not?",
     ["Compares without converting types first",
      "Compares object contents rather than references",
      "Checks that both sides are defined",
      "Works on arrays as well as primitives"],
     "Compares without converting types first",
     "`==` coerces types, so `'1' == 1` is true. `===` requires the same type, so `'1' === 1` "
     "is false. Neither compares object contents — both compare references for objects."),

    ("JavaScript", "medium", "What does `async`/`await` actually change about how code runs?",
     ["It lets you write promise-based code in sequential style without blocking the thread",
      "It runs the function on a separate CPU thread",
      "It makes asynchronous operations finish faster",
      "It converts callbacks into synchronous, blocking calls"],
     "It lets you write promise-based code in sequential style without blocking the thread",
     "`await` suspends the async function and returns control to the event loop until the "
     "promise settles. JavaScript stays single-threaded; nothing gets faster, it just reads "
     "sequentially."),

    ("JavaScript", "hard", "What is a closure, in practical terms?",
     ["A function that keeps access to variables from the scope it was created in",
      "A function that has finished executing and been garbage collected",
      "A way to make an object's properties read-only",
      "A loop construct that closes over an iterator"],
     "A function that keeps access to variables from the scope it was created in",
     "The inner function holds a live reference to the enclosing scope, which is why it still "
     "works after the outer function returns. That is also why closures can hold memory alive."),

    ("TypeScript", "medium", "What is the practical difference between `interface` and `type`?",
     ["`interface` supports declaration merging; `type` can express unions and intersections",
      "`type` is checked at runtime, `interface` only at compile time",
      "`interface` is faster to compile",
      "They are identical — `type` is deprecated"],
     "`interface` supports declaration merging; `type` can express unions and intersections",
     "Both describe shapes. Only `interface` can be reopened and merged (useful for extending "
     "library types); only `type` can alias a union such as `'a' | 'b'`. Neither exists at "
     "runtime."),

    ("TypeScript", "easy", "What does the `any` type disable?",
     ["Type checking on that value",
      "Runtime validation of the value",
      "The ability to assign the value to other variables",
      "Compilation of the file"],
     "Type checking on that value",
     "`any` opts a value out of the type system, so errors stop being caught. `unknown` is the "
     "safer alternative — it forces a narrowing check before use."),

    ("C++", "medium", "What problem do smart pointers such as `std::unique_ptr` solve?",
     ["They free memory automatically when the owner goes out of scope",
      "They make pointer arithmetic safe",
      "They allow two owners to share one allocation",
      "They move allocation from the heap to the stack"],
     "They free memory automatically when the owner goes out of scope",
     "`unique_ptr` ties the lifetime of a heap allocation to a scope (RAII), preventing leaks. "
     "Sharing is what `shared_ptr` is for — `unique_ptr` deliberately forbids it."),

    # ---------------- Databases ----------------
    ("SQL", "easy", "Which JOIN keeps every row from the left table, even with no match on the right?",
     ["LEFT JOIN", "INNER JOIN", "CROSS JOIN", "SELF JOIN"],
     "LEFT JOIN",
     "A LEFT JOIN returns all left rows, filling the right-hand columns with NULL when there "
     "is no match. An INNER JOIN drops those rows entirely."),

    ("SQL", "medium", "What is the difference between `WHERE` and `HAVING`?",
     ["`WHERE` filters rows before grouping; `HAVING` filters groups after aggregation",
      "`HAVING` is simply the newer syntax for `WHERE`",
      "`WHERE` works on numbers, `HAVING` on text",
      "`HAVING` can only be used with ORDER BY"],
     "`WHERE` filters rows before grouping; `HAVING` filters groups after aggregation",
     "You cannot put an aggregate such as `COUNT(*) > 5` in `WHERE`, because aggregation has "
     "not happened yet. That condition belongs in `HAVING`."),

    ("SQL", "medium", "What does `GROUP BY` do?",
     ["Collapses rows sharing a value into one row per group, for aggregation",
      "Sorts the result set by the named column",
      "Removes duplicate rows from the result",
      "Creates an index on the named column"],
     "Collapses rows sharing a value into one row per group, for aggregation",
     "Sorting is `ORDER BY`; de-duplicating is `DISTINCT`. `GROUP BY` exists so aggregates "
     "like SUM and COUNT can be computed per group."),

    ("SQL", "hard", "A query filtering on an indexed column is still slow. Which explanation is most likely?",
     ["A function is applied to the column, so the index cannot be used",
      "Indexes only speed up JOINs, never WHERE clauses",
      "The table has too few rows for an index to exist",
      "Indexes are ignored unless the query uses ORDER BY"],
     "A function is applied to the column, so the index cannot be used",
     "`WHERE lower(email) = ?` cannot use a plain index on `email`, because the stored values "
     "are not what is being compared. Either index the expression or restructure the query."),

    ("PostgreSQL", "medium", "What does an index cost you, in exchange for faster reads?",
     ["Slower writes and extra storage",
      "Loss of transactional guarantees",
      "A table-level lock on every query",
      "Nothing — indexes are free"],
     "Slower writes and extra storage",
     "Every INSERT, UPDATE and DELETE must also maintain the index, and the index occupies "
     "disk. That is why you index the columns you actually filter and join on, not all of them."),

    ("PostgreSQL", "hard", "What does a transaction's ACID 'Isolation' property guarantee?",
     ["Concurrent transactions do not see each other's uncommitted changes",
      "Each transaction runs on a dedicated CPU core",
      "Committed data survives a crash",
      "Every transaction either fully completes or fully rolls back"],
     "Concurrent transactions do not see each other's uncommitted changes",
     "Durability is surviving a crash; Atomicity is all-or-nothing. Isolation is specifically "
     "about concurrent transactions not interfering — the exact degree depends on the "
     "isolation level."),

    ("MongoDB", "easy", "How does MongoDB store a record?",
     ["As a document, typically with a flexible, nested structure",
      "As a row in a fixed-column table",
      "As a key-value pair with no structure",
      "As a node in a graph with typed edges"],
     "As a document, typically with a flexible, nested structure",
     "MongoDB is a document store: records are BSON documents that can nest and need not share "
     "an identical shape, which is the main trade-off against a relational table."),

    # ---------------- Backend & APIs ----------------
    ("REST APIs", "easy", "Which HTTP method is *not* idempotent?",
     ["POST", "GET", "PUT", "DELETE"],
     "POST",
     "Sending the same POST twice usually creates two resources. GET changes nothing; PUT and "
     "DELETE applied twice leave the same end state, which is what idempotent means."),

    ("REST APIs", "medium", "A client sends valid JSON but omits a required field. Which status fits best?",
     ["422 Unprocessable Content (or 400 Bad Request)",
      "500 Internal Server Error",
      "404 Not Found",
      "204 No Content"],
     "422 Unprocessable Content (or 400 Bad Request)",
     "The problem is the client's payload, so it belongs in the 4xx range. A 500 would wrongly "
     "blame the server and mask a client bug behind an alert."),

    ("REST APIs", "hard", "Why return 201 with a `Location` header instead of 200 after creating a resource?",
     ["It tells the client a resource was created and where to find it",
      "It is required for the response body to contain JSON",
      "200 is reserved for GET requests only",
      "It makes the request idempotent"],
     "It tells the client a resource was created and where to find it",
     "201 carries the specific meaning 'created', and `Location` points at the new resource so "
     "the client need not guess its URL."),

    ("Spring Boot", "medium", "What does `@RestController` add over `@Controller`?",
     ["It implies `@ResponseBody`, so return values are serialised instead of resolved as views",
      "It enables database transactions on every method",
      "It makes the class a singleton",
      "It registers the class for scheduled execution"],
     "It implies `@ResponseBody`, so return values are serialised instead of resolved as views",
     "`@RestController` is `@Controller` plus `@ResponseBody`. Without it, a returned String is "
     "treated as a view name, which is the classic 'why is my API returning 404' bug."),

    ("Spring Boot", "hard", "What is dependency injection actually buying you?",
     ["Components declare what they need and are handed it, so they can be tested in isolation",
      "Faster startup because beans load lazily",
      "Automatic database connection pooling",
      "Compile-time checking of configuration files"],
     "Components declare what they need and are handed it, so they can be tested in isolation",
     "The point is inverted control: a class that is given its collaborators can be handed test "
     "doubles, instead of constructing hard-coded dependencies it cannot replace."),

    ("Microservices", "medium", "What problem does a circuit breaker solve?",
     ["It fails fast when a downstream service is unhealthy, stopping cascading failures",
      "It encrypts traffic between services",
      "It balances load evenly across instances",
      "It rolls back distributed transactions"],
     "It fails fast when a downstream service is unhealthy, stopping cascading failures",
     "Without one, callers queue up on a dying dependency until their own threads exhaust, and "
     "the outage spreads. The breaker trips and returns an immediate error or fallback."),

    ("Microservices", "hard", "What is the main cost of splitting a monolith into microservices?",
     ["A local function call becomes a network call that can fail, time out or arrive twice",
      "You can no longer use a relational database",
      "Services must all be written in the same language",
      "Horizontal scaling becomes impossible"],
     "A local function call becomes a network call that can fail, time out or arrive twice",
     "Distribution buys independent deploys and scaling, and charges you in partial failure, "
     "latency, and consistency problems that simply did not exist in-process."),

    ("Node.js", "medium", "Why is blocking the Node.js event loop a problem?",
     ["One long synchronous operation stalls every other pending request",
      "It causes the process to run out of memory",
      "It disables asynchronous callbacks permanently",
      "It forces Node to spawn additional processes"],
     "One long synchronous operation stalls every other pending request",
     "Node handles concurrency on a single thread. A synchronous CPU-heavy loop holds that "
     "thread, so unrelated requests simply wait — the fix is to offload or chunk the work."),

    ("Express", "easy", "What is middleware in Express?",
     ["A function that runs on a request before the route handler and can pass control on",
      "A database layer between the app and storage",
      "A build step that bundles the server code",
      "A separate process that proxies requests"],
     "A function that runs on a request before the route handler and can pass control on",
     "Middleware receives `(req, res, next)` and either responds or calls `next()`. Logging, "
     "authentication and body parsing are all typically middleware."),

    # ---------------- Frontend ----------------
    ("React", "easy", "Why does React need a `key` prop on list items?",
     ["To identify which items changed, were added, or were removed between renders",
      "To set the HTML id attribute on each element",
      "To define the sort order of the list",
      "To make each item focusable for accessibility"],
     "To identify which items changed, were added, or were removed between renders",
     "Keys let React match elements across renders instead of rebuilding them. Using the array "
     "index as a key breaks this as soon as the list is reordered or filtered."),

    ("React", "medium", "What does the dependency array of `useEffect` control?",
     ["When the effect re-runs after a render",
      "Which variables the effect is allowed to read",
      "The order effects run in",
      "Whether the component re-renders at all"],
     "When the effect re-runs after a render",
     "An empty array runs it once after mount; omitting the array runs it after every render. "
     "Leaving out a value the effect reads is the usual cause of stale-closure bugs."),

    ("React", "hard", "What does `useState` actually guarantee about when state updates?",
     ["Updates are applied asynchronously and batched, so reading state right after setting it gives the old value",
      "State updates immediately and synchronously",
      "State updates only when the component unmounts",
      "Each setState call triggers exactly one separate re-render"],
     "Updates are applied asynchronously and batched, so reading state right after setting it gives the old value",
     "React batches updates and re-renders with the new value. If the next value depends on the "
     "previous one, use the function form `setCount(c => c + 1)`."),

    ("HTML", "easy", "Why use semantic elements such as `<nav>` and `<main>` instead of `<div>`?",
     ["They convey meaning to screen readers and search engines",
      "They render faster in the browser",
      "They are required for CSS to apply",
      "They automatically make the page responsive"],
     "They convey meaning to screen readers and search engines",
     "A `<div>` carries no meaning. Semantic elements expose structure to assistive technology "
     "and let a user jump directly to the main content."),

    ("CSS", "medium", "In flexbox, what does `justify-content` align along?",
     ["The main axis", "The cross axis", "Both axes equally", "The document baseline"],
     "The main axis",
     "`justify-content` works along the main axis (horizontal by default); `align-items` works "
     "along the cross axis. Changing `flex-direction` swaps which is which."),

    ("CSS", "hard", "What does `box-sizing: border-box` change?",
     ["Padding and border are counted inside the declared width instead of added to it",
      "Margins stop collapsing between adjacent elements",
      "The element is removed from normal document flow",
      "Borders are drawn outside the element's background"],
     "Padding and border are counted inside the declared width instead of added to it",
     "With the default `content-box`, a 200px element with 20px padding occupies 240px. "
     "`border-box` keeps the total at 200px, which is why most resets apply it globally."),

    # ---------------- Data, ML & AI ----------------
    ("Machine learning", "easy", "What is overfitting?",
     ["The model learns noise in the training data and generalises poorly to new data",
      "The model is too simple to capture the pattern",
      "Training takes longer than expected",
      "The dataset contains more features than rows"],
     "The model learns noise in the training data and generalises poorly to new data",
     "The signature is a large gap between training and validation performance. A model too "
     "simple to fit the pattern is underfitting, the opposite failure."),

    ("Machine learning", "medium", "Why hold out a validation set separate from the test set?",
     ["So hyperparameters can be tuned without leaking information from the final test set",
      "Because the test set is usually too small to train on",
      "To speed up training by using less data",
      "Because scikit-learn requires three splits"],
     "So hyperparameters can be tuned without leaking information from the final test set",
     "Tuning against the test set makes its score optimistic — it is no longer an unseen "
     "estimate. The validation set absorbs that tuning so the test set stays honest."),

    ("Machine learning", "medium", "On a dataset where 99% of cases are negative, why is accuracy a poor metric?",
     ["Predicting 'negative' every time already scores 99% while finding nothing",
      "Accuracy cannot be computed on imbalanced data",
      "Accuracy only applies to regression problems",
      "It requires the classes to be perfectly balanced to compute"],
     "Predicting 'negative' every time already scores 99% while finding nothing",
     "This is the classic fraud/disease detection trap. Precision, recall and the confusion "
     "matrix show what accuracy hides."),

    ("Machine learning", "hard", "What does the precision/recall trade-off mean in practice?",
     ["Catching more true positives usually means accepting more false positives",
      "Improving one metric always improves the other",
      "Precision and recall are different names for the same quantity",
      "The trade-off only exists in multi-class problems"],
     "Catching more true positives usually means accepting more false positives",
     "Lowering the decision threshold raises recall and generally lowers precision. Which "
     "matters more is a business decision: a missed cancer and a false alarm cost differently."),

    ("Deep learning", "medium", "What would a neural network without non-linear activations be equivalent to?",
     ["A single linear model, no matter how many layers it has",
      "A decision tree",
      "A model that cannot be trained at all",
      "A convolutional network"],
     "A single linear model, no matter how many layers it has",
     "Composing linear maps gives another linear map. Non-linearities such as ReLU are what "
     "let depth represent something a single layer cannot."),

    ("Deep learning", "hard", "What does dropout do during training?",
     ["Randomly deactivates units so the network cannot rely on any single one",
      "Removes training examples that produce high loss",
      "Reduces the learning rate when loss plateaus",
      "Deletes layers that are not contributing"],
     "Randomly deactivates units so the network cannot rely on any single one",
     "It is a regularisation technique that forces redundant representations. Dropout is "
     "applied during training only and switched off at inference."),

    ("NLP", "medium", "What does tokenization do?",
     ["Splits text into the units a model actually operates on",
      "Translates text into another language",
      "Removes all punctuation and stop words",
      "Encrypts text before it is stored"],
     "Splits text into the units a model actually operates on",
     "Tokens may be words, sub-words or characters. Sub-word schemes such as BPE let a fixed "
     "vocabulary represent words it has never seen."),

    ("NLP", "hard", "Why do transformers need positional encoding?",
     ["Self-attention alone has no notion of word order",
      "To compress the input to a fixed length",
      "To mask out padding tokens",
      "To reduce the vocabulary size"],
     "Self-attention alone has no notion of word order",
     "Attention treats the input as a set, so 'dog bites man' and 'man bites dog' would be "
     "identical. Positional information must be injected explicitly."),

    ("PyTorch", "medium", "What does calling `.backward()` on a loss tensor do?",
     ["Computes gradients of the loss with respect to parameters, via autograd",
      "Updates the model weights directly",
      "Reverses the order of layers for inference",
      "Frees the memory used by the forward pass"],
     "Computes gradients of the loss with respect to parameters, via autograd",
     "`.backward()` only populates `.grad`. Applying those gradients is the optimiser's job "
     "via `optimizer.step()`, and `zero_grad()` clears them between batches."),

    ("TensorFlow", "easy", "What is a tensor?",
     ["A multi-dimensional array of numbers",
      "A layer in a neural network",
      "A function that computes gradients",
      "A training loop abstraction"],
     "A multi-dimensional array of numbers",
     "Scalars, vectors and matrices are tensors of rank 0, 1 and 2. The framework adds "
     "gradient tracking and device placement on top."),

    ("scikit-learn", "hard", "Why call `fit_transform` on training data but only `transform` on test data?",
     ["Fitting on test data leaks its statistics into preprocessing and inflates the score",
      "`fit_transform` is simply faster on larger arrays",
      "Test data has a different number of columns",
      "`transform` cannot be called on training data"],
     "Fitting on test data leaks its statistics into preprocessing and inflates the score",
     "A scaler must learn mean and variance from training data alone. Refitting on test data "
     "lets information from the held-out set reach the model — a classic leakage bug."),

    ("Pandas", "easy", "What is the difference between `.loc` and `.iloc`?",
     ["`.loc` selects by label; `.iloc` selects by integer position",
      "`.loc` returns a copy; `.iloc` returns a view",
      "`.loc` works on rows only; `.iloc` on columns only",
      "`.iloc` is the deprecated alias of `.loc`"],
     "`.loc` selects by label; `.iloc` selects by integer position",
     "On a DataFrame with a non-integer index the distinction matters: `.loc['b']` finds the "
     "row labelled 'b', `.iloc[1]` finds the second row whatever it is labelled."),

    ("Pandas", "medium", "What does `df.groupby('city')['salary'].mean()` return?",
     ["The average salary for each city",
      "The total salary across all cities",
      "A DataFrame sorted by city",
      "The rows where salary is above the mean"],
     "The average salary for each city",
     "`groupby` splits the rows by city, then the aggregation is applied within each group — "
     "the split-apply-combine pattern."),

    ("NumPy", "medium", "What does broadcasting let you do?",
     ["Operate on arrays of different shapes without writing explicit loops",
      "Send an array across a network to worker nodes",
      "Convert an array to a Python list efficiently",
      "Share one array between multiple processes"],
     "Operate on arrays of different shapes without writing explicit loops",
     "Adding a shape-(3,) array to a shape-(4,3) array applies it to every row. The smaller "
     "array is virtually expanded; no copy is actually materialised."),

    ("Statistics", "hard", "What does a p-value of 0.03 actually mean?",
     ["If the null hypothesis were true, data this extreme would occur 3% of the time",
      "There is a 3% probability the null hypothesis is true",
      "There is a 97% probability the result is real",
      "The effect size is 3% of the mean"],
     "If the null hypothesis were true, data this extreme would occur 3% of the time",
     "This is the most common misreading in statistics. A p-value is computed *assuming* the "
     "null is true, so it cannot be the probability that the null is true. It also says "
     "nothing about effect size."),

    ("Statistics", "medium", "Why does correlation not imply causation?",
     ["A third variable, or reverse causation, can produce the same association",
      "Correlation coefficients are unreliable below 1000 samples",
      "Correlation only measures non-linear relationships",
      "Causation requires a correlation above 0.9"],
     "A third variable, or reverse causation, can produce the same association",
     "Ice cream sales and drownings correlate because both rise with temperature. "
     "Establishing causation needs a controlled experiment or a careful causal design."),

    ("Data visualisation", "medium", "When is truncating the y-axis so it does not start at zero misleading?",
     ["On a bar chart, because bar length is how the reader judges magnitude",
      "On a line chart showing a trend over time",
      "Whenever the data has more than two series",
      "Only when the chart is printed rather than on screen"],
     "On a bar chart, because bar length is how the reader judges magnitude",
     "Bars encode value by length from zero, so cutting the axis exaggerates differences. "
     "Line charts encode change by slope, so a truncated axis is often legitimate there."),

    ("Power BI", "easy", "What is a measure in Power BI?",
     ["A calculation evaluated at query time against the current filter context",
      "A static column stored in the source table",
      "A visual element on the report canvas",
      "A connection string to a data source"],
     "A calculation evaluated at query time against the current filter context",
     "A measure recomputes as the user slices and filters; a calculated column is computed once "
     "at refresh and stored per row."),

    ("Excel", "easy", "Why does `VLOOKUP` fail when the lookup column is not the leftmost?",
     ["VLOOKUP can only search the first column of its range and return values to the right",
      "VLOOKUP requires the data to be sorted alphabetically",
      "VLOOKUP cannot handle text values",
      "VLOOKUP is limited to 1000 rows"],
     "VLOOKUP can only search the first column of its range and return values to the right",
     "This is exactly why `INDEX/MATCH` and the newer `XLOOKUP` exist — they can look left and "
     "do not break when columns are inserted."),

    # ---------------- Cloud & DevOps ----------------
    ("Docker", "easy", "What is the difference between an image and a container?",
     ["An image is the read-only template; a container is a running instance of it",
      "An image runs; a container stores the filesystem",
      "They are the same thing under different commands",
      "An image holds data; a container holds code"],
     "An image is the read-only template; a container is a running instance of it",
     "One image can back many containers. Changes inside a running container live in its "
     "writable layer and vanish unless committed or stored in a volume."),

    ("Docker", "medium", "Why copy `requirements.txt` and install dependencies *before* copying application code?",
     ["So the dependency layer stays cached when only application code changes",
      "Because Docker cannot copy multiple files in one command",
      "To reduce the number of layers in the final image",
      "Because pip must run before any COPY instruction"],
     "So the dependency layer stays cached when only application code changes",
     "Docker caches layers in order and invalidates everything after the first change. Copying "
     "code first would reinstall every dependency on every edit."),

    ("Kubernetes", "medium", "What is a Pod?",
     ["The smallest deployable unit — one or more containers sharing a network namespace and storage",
      "A physical machine in the cluster",
      "A load balancer in front of a service",
      "A snapshot of cluster configuration"],
     "The smallest deployable unit — one or more containers sharing a network namespace and storage",
     "Containers in a pod share an IP and can reach each other on localhost. A machine is a "
     "Node; stable networking in front of pods is a Service."),

    ("Kubernetes", "hard", "What does a readiness probe do that a liveness probe does not?",
     ["It removes the pod from service endpoints without restarting it",
      "It restarts the container when it fails",
      "It runs only once at container startup",
      "It checks resource limits rather than health"],
     "It removes the pod from service endpoints without restarting it",
     "Liveness failure restarts the container; readiness failure just stops traffic being sent "
     "to it. Using liveness where readiness belongs causes needless restart loops during "
     "warm-up."),

    ("AWS", "easy", "What is Amazon S3 best suited to?",
     ["Object storage for files such as images, backups and static assets",
      "Running virtual machines",
      "Hosting a relational database",
      "Managing DNS records"],
     "Object storage for files such as images, backups and static assets",
     "S3 stores objects in buckets. EC2 runs virtual machines, RDS hosts relational databases, "
     "and Route 53 handles DNS."),

    ("AWS", "medium", "Why use an IAM role rather than embedding access keys in your application?",
     ["Credentials are issued temporarily and rotated automatically, with nothing to leak in code",
      "Roles are cheaper than access keys",
      "Access keys cannot grant S3 permissions",
      "Roles remove the need for any permission policy"],
     "Credentials are issued temporarily and rotated automatically, with nothing to leak in code",
     "A hard-coded key committed to a repository is one of the most common real breaches. A "
     "role attached to the compute resource avoids a long-lived secret existing at all."),

    ("CI/CD", "medium", "What is the main point of continuous integration?",
     ["Changes are merged and tested frequently, so breakage is caught while it is small",
      "Code is deployed to production on every commit",
      "Developers work on long-lived branches in isolation",
      "Tests are run manually before each release"],
     "Changes are merged and tested frequently, so breakage is caught while it is small",
     "CI is about integrating often and verifying automatically. Deploying every change to "
     "production is continuous *deployment*, a separate decision."),

    ("Git", "easy", "What does `git merge` do that `git rebase` does not?",
     ["It creates a merge commit, preserving the original branch history",
      "It deletes the source branch after combining",
      "It discards conflicting changes automatically",
      "It uploads the result to the remote"],
     "It creates a merge commit, preserving the original branch history",
     "Rebase replays your commits onto a new base, producing linear history but rewriting "
     "commit hashes — which is why rebasing shared branches causes trouble."),

    ("Git", "hard", "Why is `git push --force` dangerous on a shared branch?",
     ["It overwrites remote history, discarding commits teammates may already have",
      "It permanently deletes the local repository",
      "It bypasses the test suite",
      "It merges without checking for conflicts"],
     "It overwrites remote history, discarding commits teammates may already have",
     "`--force-with-lease` is the safer form: it refuses to push when the remote has moved "
     "since you last fetched."),

    ("Terraform", "medium", "What is the Terraform state file for?",
     ["Mapping declared resources to the real infrastructure that was created",
      "Storing provider credentials securely",
      "Caching module downloads",
      "Recording a log of past terraform commands"],
     "Mapping declared resources to the real infrastructure that was created",
     "Without state, Terraform cannot tell what it already made and would recreate everything. "
     "State can contain secrets, so it belongs in a secured remote backend, not in Git."),

    ("Jenkins", "easy", "What is a Jenkins pipeline?",
     ["A defined sequence of build, test and deploy stages, usually stored as code in the repository",
      "A queue of agents waiting for work",
      "A plugin that monitors server health",
      "A database of past build artefacts"],
     "A defined sequence of build, test and deploy stages, usually stored as code in the repository",
     "A `Jenkinsfile` keeps the pipeline versioned with the code it builds, so the build process "
     "changes through review like any other change."),

    # ---------------- Testing & QA ----------------
    ("PyTest", "medium", "What is a pytest fixture for?",
     ["Providing reusable setup and teardown that tests request by name",
      "Asserting that two values are equal",
      "Marking a test as expected to fail",
      "Running tests in parallel"],
     "Providing reusable setup and teardown that tests request by name",
     "A fixture's scope (function, module, session) controls how often it is rebuilt — useful "
     "when setup such as a database is expensive."),

    ("JUnit", "easy", "What does a unit test aim to verify?",
     ["One unit of behaviour in isolation from its dependencies",
      "That the whole application starts correctly",
      "That the user interface renders as designed",
      "That the system handles production load"],
     "One unit of behaviour in isolation from its dependencies",
     "Isolation is what makes a failure point at a specific cause. Testing components together "
     "is integration testing; load behaviour is performance testing."),

    ("Test automation", "medium", "What makes a test 'flaky'?",
     ["It passes or fails non-deterministically without the code changing",
      "It takes a long time to run",
      "It tests more than one function",
      "It was written after the code rather than before"],
     "It passes or fails non-deterministically without the code changing",
     "Usually timing, test order dependence, or shared external state. Flaky tests are "
     "corrosive because they train the team to ignore red builds."),

    ("Selenium", "medium", "Why are explicit waits preferred over `sleep()` in UI tests?",
     ["They continue as soon as the condition is met, instead of always pausing a fixed time",
      "They are the only way to locate elements by CSS selector",
      "They run the browser in headless mode",
      "They retry the entire test on failure"],
     "They continue as soon as the condition is met, instead of always pausing a fixed time",
     "A fixed sleep is either too short (flaky) or too long (slow). An explicit wait polls for "
     "the actual condition up to a timeout."),

    ("API testing", "medium", "Why should an API test assert on the status code *and* the response body?",
     ["A 200 response can still contain wrong or missing data",
      "Status codes are not reliable across frameworks",
      "The body is required to parse the status code",
      "Assertions on status codes are deprecated"],
     "A 200 response can still contain wrong or missing data",
     "Checking only the status confirms the endpoint answered, not that it answered correctly — "
     "which is the thing the test is supposed to protect."),

    # ---------------- Tooling, practice & design ----------------
    ("Agile", "easy", "What is the purpose of a sprint retrospective?",
     ["The team inspects how it worked and agrees what to change next sprint",
      "The team demonstrates completed work to stakeholders",
      "The team estimates upcoming backlog items",
      "The manager reviews individual performance"],
     "The team inspects how it worked and agrees what to change next sprint",
     "Demonstrating work is the sprint review; estimating is refinement or planning. The "
     "retrospective is specifically about the process, not the product."),

    ("System design", "medium", "What does horizontal scaling mean?",
     ["Adding more machines and distributing load across them",
      "Adding more CPU and memory to an existing machine",
      "Splitting a database into more tables",
      "Caching responses closer to the user"],
     "Adding more machines and distributing load across them",
     "Upgrading one machine is vertical scaling, which eventually hits a hardware ceiling. "
     "Horizontal scaling requires the application to handle shared state carefully."),

    ("System design", "hard", "What is the main trade-off a cache introduces?",
     ["Reads get faster, but served data can be stale and invalidation is hard to get right",
      "Writes get faster, but reads become slower",
      "Storage cost falls, but network cost rises",
      "Security improves, but availability drops"],
     "Reads get faster, but served data can be stale and invalidation is hard to get right",
     "The hard question is always when to evict or refresh. A cache that is never invalidated "
     "correctly serves confidently wrong answers."),

    ("System design", "hard", "Why might a system choose eventual consistency over strong consistency?",
     ["To stay available and responsive during network partitions, accepting temporary staleness",
      "Because eventual consistency is simpler to reason about",
      "Because strong consistency cannot be implemented across multiple machines",
      "To reduce the amount of data stored"],
     "To stay available and responsive during network partitions, accepting temporary staleness",
     "This is the practical reading of the CAP theorem: when a partition happens you choose "
     "consistency or availability. A social feed can tolerate staleness; a bank balance cannot."),

    ("Problem solving", "medium", "A bug appears in production but not in staging. What is the most useful first step?",
     ["Identify what actually differs between the two environments",
      "Rewrite the affected module from scratch",
      "Roll back the last ten deployments",
      "Add more logging everywhere and wait"],
     "Identify what actually differs between the two environments",
     "Data volume, configuration, credentials and traffic shape are the usual culprits. "
     "Narrowing the difference is what makes the bug reproducible, and reproducible is "
     "most of the fix."),

    ("Code review", "medium", "What is the most valuable thing a code review catches?",
     ["Design and correctness problems a linter or formatter cannot see",
      "Inconsistent indentation and naming style",
      "Compilation errors",
      "Slow test execution"],
     "Design and correctness problems a linter or formatter cannot see",
     "Formatting should be automated so reviewers spend attention on logic, edge cases and "
     "whether the approach fits the system."),

    ("Communication", "medium", "You will miss a deadline you committed to. When is it best to say so?",
     ["As soon as you are confident it will slip, with a revised estimate",
      "On the deadline itself, once it is certain",
      "After the deadline, with the completed work",
      "Only if someone asks for a status update"],
     "As soon as you are confident it will slip, with a revised estimate",
     "Early warning is what lets others re-plan. The cost of a late surprise is almost always "
     "higher than the cost of an early, honest estimate."),
]

BANK_SIZE = len(BANK)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _by_skill() -> Dict[str, List[Item]]:
    index: Dict[str, List[Item]] = {}
    for item in BANK:
        index.setdefault(item[0], []).append(item)
    return index


SKILL_INDEX = _by_skill()
COVERED_SKILLS = sorted(SKILL_INDEX)


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


def _target_counts(total: int, target: str) -> Dict[str, int]:
    """Split `total` questions across difficulties, never losing one to rounding."""
    counts: Dict[str, int] = {}
    assigned = 0
    mix = MIX_BY_TARGET.get(target, MIX_BY_TARGET["medium"])
    for level, share in mix[:-1]:
        n = round(total * share)
        counts[level] = n
        assigned += n
    counts[mix[-1][0]] = max(0, total - assigned)
    return counts


def build_mcq_test(candidate: Dict[str, Any], job: Dict[str, Any],
                   total: int = 10) -> Dict[str, Any]:
    """Build an MCQ screening test for this candidate against this role.

    Questions are drawn from skills the role requires, preferring those the resume
    evidences (verifying a claim) over those it does not (establishing a baseline).
    """
    result = match(candidate, job)
    name = candidate.get("name") or "Candidate"
    rng = random.Random(_seed(name, str(job.get("title", "")), str(total)))

    seniority = (candidate.get("seniority")
                 or (candidate.get("experience") or {}).get("seniority") or "Unclassified")
    target = DIFFICULTY_BY_SENIORITY.get(seniority, "medium")

    held = {d["skill"] for d in result["skill_detail"] if d["credit"] >= 1.0}
    required = [d["skill"] for d in result["skill_detail"]]
    # claimed skills first — an MCQ is strongest as a check on something asserted
    ordered = ([s for s in required if s in held and s in SKILL_INDEX]
               + [s for s in required if s not in held and s in SKILL_INDEX])
    testable = list(dict.fromkeys(ordered))

    wanted = _target_counts(total, target)
    picked: List[Item] = []
    used_questions = set()

    # round-robin across skills so one skill cannot dominate the test
    for level in ("hard", "medium", "easy"):
        need = wanted.get(level, 0)
        if need <= 0:
            continue
        pool_by_skill = {s: [i for i in SKILL_INDEX[s]
                             if i[1] == level and i[2] not in used_questions]
                         for s in testable}
        pool_by_skill = {s: v for s, v in pool_by_skill.items() if v}
        skills_cycle = sorted(pool_by_skill, key=lambda s: (testable.index(s),))
        while need > 0 and pool_by_skill:
            progressed = False
            for skill in list(skills_cycle):
                if need <= 0:
                    break
                bucket = pool_by_skill.get(skill)
                if not bucket:
                    continue
                item = bucket.pop(rng.randrange(len(bucket)))
                picked.append(item)
                used_questions.add(item[2])
                need -= 1
                progressed = True
                if not bucket:
                    pool_by_skill.pop(skill, None)
            if not progressed:
                break

    # if a difficulty ran dry, backfill from any level rather than return a short test
    if len(picked) < total:
        spare = [i for s in testable for i in SKILL_INDEX[s] if i[2] not in used_questions]
        rng.shuffle(spare)
        for item in spare:
            if len(picked) >= total:
                break
            picked.append(item)
            used_questions.add(item[2])

    rng.shuffle(picked)
    questions = [_render(item, name, rng) for item in picked]
    seconds = sum(SECONDS_PER_QUESTION.get(q["difficulty_level"], 70) for q in questions)

    return {
        "candidate": name,
        "candidate_id": candidate.get("candidate_id"),
        "job_title": job.get("title"),
        "job_id": job.get("job_id"),
        "seniority": seniority,
        "target_difficulty": target,
        "questions": questions,
        "requested": total,
        "coverage": {
            "role_skills": len(required),
            "testable": len(testable),
            "asked": len(questions),
            "note": ("full coverage" if len(questions) >= total else
                     f"the question bank covers {len(testable)} of this role's "
                     f"{len(required)} skills, so the test is {len(questions)} questions"),
        },
        "mix": {lvl: sum(1 for q in questions if q["difficulty_level"] == lvl)
                for lvl in ("easy", "medium", "hard")},
        "time_limit_seconds": seconds,
        "pass_mark": PASS_MARK,
    }


def _render(item: Item, name: str, rng: random.Random) -> Dict[str, Any]:
    """Shuffle the options and record the correct answer by its new position."""
    skill, difficulty, text, options, correct, explanation = item
    shuffled = list(options)
    random.Random(_seed(name, text)).shuffle(shuffled)
    category = CATEGORIES.get((TAXONOMY.get(skill) or ("tool",))[0], "Tooling & practice")
    return {
        "question_id": "mcq_" + hashlib.sha1(text.encode()).hexdigest()[:10],
        "question_text": text,
        "options": shuffled,
        "correct_index": shuffled.index(correct),
        "skill": skill,
        "skill_category": category,
        "difficulty_level": difficulty,
        "explanation": explanation,
        "seconds": SECONDS_PER_QUESTION.get(difficulty, 70),
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_mcq_test(test: Dict[str, Any], answers: Sequence[Optional[int]]) -> Dict[str, Any]:
    """Mark a completed test. `answers` holds the chosen option index per question,
    or None where the candidate did not answer.

    Harder questions are worth more, because scoring a hard and an easy question
    identically would let a candidate pass on easy items alone.
    """
    weights = {"easy": 1.0, "medium": 1.5, "hard": 2.0}
    questions = test["questions"]
    rows: List[Dict[str, Any]] = []
    earned = possible = 0.0
    by_skill: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}

    for i, q in enumerate(questions):
        given = answers[i] if i < len(answers) else None
        correct = given is not None and int(given) == q["correct_index"]
        weight = weights.get(q["difficulty_level"], 1.0)
        possible += weight
        if correct:
            earned += weight

        for bucket, key in ((by_skill, q["skill"]), (by_category, q["skill_category"])):
            slot = bucket.setdefault(key, {"asked": 0, "correct": 0})
            slot["asked"] += 1
            slot["correct"] += 1 if correct else 0

        rows.append({
            "question_id": q["question_id"], "question_text": q["question_text"],
            "skill": q["skill"], "difficulty_level": q["difficulty_level"],
            "options": q["options"],
            "given_index": None if given is None else int(given),
            "given_text": None if given is None else q["options"][int(given)],
            "correct_index": q["correct_index"],
            "correct_text": q["options"][q["correct_index"]],
            "correct": correct, "answered": given is not None,
            "explanation": q["explanation"],
        })

    score = round(earned / possible * 100, 1) if possible else 0.0
    right = sum(1 for r in rows if r["correct"])
    unanswered = sum(1 for r in rows if not r["answered"])

    def pct(slot: Dict[str, int]) -> float:
        return round(slot["correct"] / slot["asked"] * 100, 1) if slot["asked"] else 0.0

    skill_pct = {k: pct(v) for k, v in sorted(by_skill.items())}
    weak = sorted([k for k, v in by_skill.items() if pct(v) < 50 and v["asked"] >= 1])
    strong = sorted([k for k, v in by_skill.items() if pct(v) == 100.0])

    if not rows:
        verdict, recommendation = "no questions", "no test was taken"
    elif score >= 85:
        verdict, recommendation = "strong", "clear technical pass — proceed to the interview"
    elif score >= PASS_MARK:
        verdict, recommendation = "pass", "meets the screening bar — probe the weak areas at interview"
    elif score >= 40:
        verdict, recommendation = "borderline", "below the bar — a second opinion is warranted before rejecting"
    else:
        verdict, recommendation = "fail", "does not meet the technical screening bar for this role"

    return {
        "candidate": test.get("candidate"), "candidate_id": test.get("candidate_id"),
        "job_title": test.get("job_title"), "job_id": test.get("job_id"),
        "score": score, "pass_mark": PASS_MARK, "passed": score >= PASS_MARK,
        "verdict": verdict, "recommendation": recommendation,
        "correct": right, "of": len(rows), "unanswered": unanswered,
        "weighted_points": f"{round(earned, 1)}/{round(possible, 1)}",
        "by_skill": skill_pct,
        "by_category": {k: pct(v) for k, v in sorted(by_category.items())},
        "weak_skills": weak, "strong_skills": strong,
        "answers": rows,
        "summary": _summary(test.get("candidate"), score, right, len(rows), unanswered, weak),
    }


def _summary(name: Optional[str], score: float, right: int, total: int,
             unanswered: int, weak: List[str]) -> str:
    who = name or "The candidate"
    parts = [f"{who} scored {score}% ({right} of {total} correct"]
    parts.append(f", {unanswered} unanswered)" if unanswered else ")")
    if weak:
        shown = ", ".join(weak[:4])
        parts.append(f". Weakest on {shown}" + (" and others" if len(weak) > 4 else ""))
        parts.append(" — worth probing directly at interview.")
    else:
        parts.append(". No skill fell below half marks.")
    return "".join(parts)


def bank_stats() -> Dict[str, Any]:
    """What the bank actually covers — used by the UI and the tests."""
    per_level: Dict[str, int] = {}
    for _, difficulty, *_ in BANK:
        per_level[difficulty] = per_level.get(difficulty, 0) + 1
    per_category: Dict[str, int] = {}
    for skill, *_ in BANK:
        cat = CATEGORIES.get((TAXONOMY.get(skill) or ("tool",))[0], "Tooling & practice")
        per_category[cat] = per_category.get(cat, 0) + 1
    return {"questions": BANK_SIZE, "skills": len(SKILL_INDEX),
            "by_difficulty": per_level, "by_category": dict(sorted(per_category.items())),
            "covered_skills": COVERED_SKILLS}
