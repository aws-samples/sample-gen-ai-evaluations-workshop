# Automated reasoning

Bedrock provides an [automated reasoning capability](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html). This capability, which is unique to Bedrock, translates a formal policy document into an automated reasoning policy. The policy contains rules extracted from the document. At runtime, these rules are evaluated against the model's output.

For example, consider the PDF document `housing-code.pdf`, which is part of the San Francisco housing code. Page four of this document has this rule:

```
(5) Shower heads with no more than a three-gallon-per-minute flow shall be required.
Showerheads of the ball-joint type that cannot easily be removed from the wall without structural
alteration are exempt from this requirement.
```

The automated reasoning policy contains this equivalent rule:

```
showerheadFlowRate is no more than 3 or isBallJointShowerhead is true
```

You can attach an automated reasoning policy to a guardrail, and it will provide several findings based on the policy. These findings will show any extracted premises and claims - in other words, parts of the model output that map to the policy rules. If possible, the finding will also display whether the claim is valid or not, with a confidence score. Note however that these findings are only advisory; the guardrail will not "fail" based on these findings.

Using automated reasoning policies requires careful inspection of the generated rules with the help of a reasoning expert. For that reason, we do not provide an example notebook here, although you can find several in the [samples repository](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/responsible_ai/bedrock-automated-reasoning-checks). If you are interested in automated reasoning policies, we encourage you to upload the `housing-code.pdf` file in the Bedrock console, examine the generated policy, and test it with sample prompts.