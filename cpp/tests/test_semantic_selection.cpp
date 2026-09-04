#include "semantic_selection.hpp"

int main()
{
    SemanticClassSelection selection = select_semantic_class(
        0.9F, 0.8F, 0.2F, 0.9F,
        0.5F, 0.5F, 0.5F, 0.5F);
    if (selection.class_label != 0 || selection.mask_probability != 0.8F)
    {
        return 1;
    }

    selection = select_semantic_class(
        0.2F, 0.9F, 0.8F, 0.7F,
        0.5F, 0.5F, 0.5F, 0.5F);
    if (selection.class_label != 1 || selection.mask_probability != 0.7F)
    {
        return 2;
    }

    selection = select_semantic_class(
        0.9F, 0.9F, 0.9F, 0.8F,
        0.5F, 0.5F, 0.5F, 0.5F);
    if (selection.class_label != 1)
    {
        return 3;
    }

    selection = select_semantic_class(
        0.9F, 0.6F, 0.9F, 0.6F,
        0.5F, 0.7F, 0.5F, 0.5F);
    if (selection.class_label != 1)
    {
        return 4;
    }

    selection = select_semantic_class(
        0.4F, 0.9F, 0.9F, 0.4F,
        0.5F, 0.5F, 0.5F, 0.5F);
    if (selection.class_label != 255 || selection.mask_probability != 0.0F)
    {
        return 5;
    }

    return 0;
}
