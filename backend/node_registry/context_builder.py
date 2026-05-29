# node for the context builder

# llms needs context to funciton effectively

# three types of content: user(messages), documents(rag), as well as environemnt(tools, user documents, ground truth for user added comments)


def context_builder_node(state: AgentState):
    pass


# context_builder -> plan -> rag(if necessary) -> tools(if necessary) -> reflect -> (loop back if necessary) -> synthesize -> finish
