use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::quote;
use syn::{Error, Fields, Ident, ItemStruct, parse_macro_input, parse_quote};

/// Creates fields and implementation automatically
#[proc_macro_attribute]
pub fn static_game_object(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Initially,
    // this macros was
    // like
    // automatically
    // filler for
    // structures, but
    // now this macros
    // useless
    let mut item = parse_macro_input!(item as ItemStruct);

    let found_crate =
        crate_name("snake-engine").expect("snake-engine must be present in Cargo.toml");
    let crate_ident = match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( ::#ident )
        }
    };

    if let Fields::Named(ref mut fields) = item.fields {
        fields
            .named
            .push(parse_quote!(pub shape: #crate_ident::geom::shapes::Shapes));
    } else {
        return Error::new_spanned(
            &item,
            "Current attribute supports structs with named fields",
        )
        .to_compile_error()
        .into();
    }

    item.attrs
        .retain(|attr| !attr.path().is_ident("game_object"));

    let class = &item.ident;

    let expanded = quote! {
        #item

        impl #crate_ident::RenderGameObject for #class {
            fn shape(&self) -> #crate_ident::geom::shapes::Shapes {
               self.shape.clone()
            }
        }
    };

    TokenStream::from(expanded)
}

/// Creates fields and implementation automatically
#[proc_macro_attribute]
pub fn text_object(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as ItemStruct);

    let found_crate =
        crate_name("snake-engine").expect("snake-engine must be present in Cargo.toml");
    let crate_ident = match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( ::#ident )
        }
    };

    if let Fields::Named(ref mut fields) = item.fields {
        fields
            .named
            .push(parse_quote!(pub info: #crate_ident::text::sprite_text::SpriteTextCreateInfo));
    } else {
        return Error::new_spanned(
            &item,
            "Current attribute supports structs with named fields",
        )
        .to_compile_error()
        .into();
    }

    item.attrs
        .retain(|attr| !attr.path().is_ident("game_object"));

    let class = &item.ident;

    let expanded = quote! {
        #item

        impl #crate_ident::RenderText for #class {
            fn info(&self) -> #crate_ident::text::sprite_text::SpriteTextCreateInfo {
                self.info.clone()
            }
        }
    };

    TokenStream::from(expanded)
}
