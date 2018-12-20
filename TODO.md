# Refactoring

## Style
Mapping out functional vs. OO parts

## Extensability
When refactoring, keep in mind the planned functionality
Possible changes to the algorithm, interactive switching for evaluation?

# Performance

## I/O
Pre-fetching and caching?

## Processing
Not an issue for now, after IO issues are solved perhaps

# Functionality

## Select section to count
The ability to trace an area to investigate
Can we do this within OpenCV GUI library?

## Automatic detection of interesting areas
Far future - How to segment into tumor/non-tumor

# Correctness
## Investigate false negative/positive rate
It seems we have many false negatives, and some false positives
This needs to be quantified somehow

A system test suite needs to be made for automatic verification

## Use HSD color space
Potentially better differentiation than HSV
https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4%3C275::AID-CYTO5%3E3.0.CO;2-8
- Add support for this in OpenCV?
